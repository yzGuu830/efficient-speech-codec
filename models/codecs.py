import torch
import torch.nn as nn
from einops import rearrange

from models.esc import BaseAudioCodec, CrossScaleRVQ, SwinTEncoder
from modules import ConvolutionLayer, TransformerLayer, Convolution2D, PatchEmbed, PatchDeEmbed

class RVQCodecs(BaseAudioCodec):
    """
    Frequency Codec with conv/swinT backbones along with RVQs
    Experiment: RVQ + CNN / RVQ + swinT
    """
    def __init__(self, 
                in_dim: int=2, 
                in_freq: int=192, 
                h_dims: list=[45,72,96,144,192,384], # [16,16,24,24,32,64]
                max_streams: int=6,   # max layer depth here
                backbone: str="conv", # swinT
                kernel_size: list=[5,2],
                conv_depth: int=1,
                patch_size: list=[3,2],
                swin_heads: list=[3,6,12,24,24], 
                swin_depth: int=2, 
                window_size: int=4, 
                mlp_ratio: float=4.,
                overlap: int=4,
                num_rvqs: int=6,
                group_size: int=3,
                codebook_dim: int=8,
                codebook_size: int=1024,
                l2norm: bool=True,
                win_len: int = 20, 
                hop_len: int = 5, 
                sr: int = 16000) -> None:
        super().__init__(in_dim, in_freq, h_dims, max_streams, win_len, hop_len, sr)

        self.quantizers = self.init_residual_vqs(patch_size, overlap, group_size, num_rvqs, codebook_dim, codebook_size, l2norm, backbone)

        if backbone=="swinT":
            self.encoder = SwinTEncoder(in_freq, in_dim, h_dims, patch_size, swin_heads, swin_depth, window_size, mlp_ratio)
            self.decoder = SwinTDecoder(in_freq, in_dim, h_dims[::-1], patch_size, swin_heads[::-1], swin_depth, window_size, mlp_ratio)
        elif backbone=="conv":
            self.encoder = ConvEncoder(in_dim, h_dims, tuple(patch_size), tuple(kernel_size), conv_depth)
            self.decoder = ConvDecoder(in_dim, h_dims[::-1], tuple(patch_size), tuple(kernel_size), conv_depth) 
        else:
            raise ValueError("backbone argument should be either `swinT` or `conv`")

    def forward_one_step(self, x, x_feat=None, num_streams=6, freeze_codebook=False):
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = rearrange(x_feat, "b h w c -> b c h w") 

        enc_hs, H,W = self.encoder(x_feat)

        z_e = enc_hs[-1]
        outputs, losses = self.quantizers(z_e, num_streams, freeze=freeze_codebook)
        cm_loss, cb_loss = losses
        z_q, codes = outputs

        recon_feat = self.decoder(z_q, (H,W))
        recon_x = self.audio_reconstruct(recon_feat)

        return {
                "cm_loss": cm_loss,
                "cb_loss": cb_loss,
                "raw_audio": x,
                "recon_audio": recon_x,
                "raw_feat": x_feat, 
                "recon_feat": recon_feat,
                "codes": codes
            }

    def forward(self, x, x_feat, num_streams, freeze_codebook=False):
        """ Forward Function.
        Args: 
            x: audio Tensor
            x_feat: audio complex STFT
            num_streams: number of streams transmitted
            train: set False in inference
            freeze_codebook: boolean set True only during pretraining stage (meanwhile streams has to be max_stream)
        """
        return self.forward_one_step(x, x_feat, num_streams, freeze_codebook)


class CSVQConvCodec(BaseAudioCodec):
    """
    Frequency Codec with conv backbones along with CSVQ
    Experiment: CSVQ + CNN 
    """
    def __init__(self, 
                 in_dim: int, 
                 in_freq: int, 
                 h_dims: list, 
                 max_streams: int, 
                 kernel_size: list=[5,2],
                 conv_depth: int=1,
                 patch_size: list=[3,2],
                 overlap: int=4,
                 group_size: int=3,
                 codebook_dim: int=8,
                 codebook_size: int=1024,
                 l2norm: bool=True,
                 init_method: str="kmeans",
                 win_len: int = 20, 
                 hop_len: int = 5, 
                 sr: int = 16000) -> None:
        super().__init__(in_dim, in_freq, h_dims, max_streams, win_len, hop_len, sr)

        self.quantizers = self.init_product_vqs(
                    patch_size=patch_size, overlap=overlap, group_size=group_size, 
                    codebook_dims=[codebook_dim]*max_streams, codebook_size=codebook_size, 
                    l2norm=l2norm, init_method=init_method,
                )
        self.encoder = ConvEncoder(in_dim, h_dims, tuple(patch_size), tuple(kernel_size), conv_depth)
        self.decoder = CrossScaleConvDecoder(in_dim, h_dims[::-1], tuple(patch_size), tuple(kernel_size), conv_depth)

    def forward_one_step(self, x, x_feat=None, num_streams=6, freeze_codebook=False):
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = rearrange(x_feat, "b h w c -> b c h w") 

        enc_hs, H,W = self.encoder(x_feat)
        recon_feat, _, codes, cm_loss, cb_loss = self.decoder(enc_hs, num_streams, self.quantizers, freeze_codebook)
        recon_x = self.audio_reconstruct(recon_feat)

        return {
                "cm_loss": cm_loss,
                "cb_loss": cb_loss,
                "raw_audio": x,
                "recon_audio": recon_x,
                "raw_feat": x_feat, 
                "recon_feat": recon_feat,
                "codes": codes
            }

    def forward(self, x, x_feat, num_streams, freeze_codebook=False):
        """ Forward Function.
        Args: 
            x: audio Tensor
            x_feat: audio complex STFT
            num_streams: number of streams transmitted
            train: set False in inference
            freeze_codebook: boolean set True only during pretraining stage (meanwhile streams has to be max_stream)
        """
        num_streams = self.max_streams if freeze_codebook else num_streams
        return self.forward_one_step(x, x_feat, num_streams, freeze_codebook)


class ConvEncoder(nn.Module):
    def __init__(self, in_dim:int=2, h_dims:list=[16,16,24,24,32,64], patch_size:tuple=(3,2), kernel_size:tuple=(5,2), depth:int=1) -> None:
        super().__init__()

        self.patch_embed = PatchEmbed(192, in_dim, patch_size, embed_dim=h_dims[0], norm_layer=False, backbone="conv")

        ins, outs = [h_dims[0]] + h_dims, h_dims
        self.pre_conv = Convolution2D(ins[0], outs[0], kernel_size, scale=False)

        blocks = nn.ModuleList()
        for i in range(1, len(h_dims)):
            blocks.append(
                ConvolutionLayer(ins[i], outs[i], depth, kernel_size,)
            )
        self.enc_blks = blocks

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pre_conv(x)

        enc_hs = [x]
        for blk in self.enc_blks:
            x = blk(x)
            enc_hs.append(x)
        
        return enc_hs, x.size(2), x.size(3)


class ConvDecoder(nn.Module):
    def __init__(self, in_dim:int=2, h_dims:list=[64,32,24,24,16,16], patch_size:tuple=(3,2), kernel_size:tuple=(5,2), depth:int=1) -> None:
        super().__init__()
        ins, outs = h_dims, h_dims[1:] + [h_dims[-1]]
    
        blocks = nn.ModuleList()
        for i in range(len(h_dims)-1):
            blocks.append(
                ConvolutionLayer(ins[i], outs[i], depth, kernel_size, transpose=True)
            )
        self.dec_blks = blocks
        self.post_conv = Convolution2D(ins[-1], outs[-1], kernel_size, scale=False)
        self.patch_deembed = PatchDeEmbed(192, in_dim, patch_size, h_dims[-1], backbone="conv")

    def forward(self, z_q, feat_shape):
    
        for blk in self.dec_blks:
            z_q = blk(z_q)

        recon_x = self.post_conv(z_q)
        recon_x = self.patch_deembed(recon_x)
        return recon_x


class CrossScaleConvDecoder(CrossScaleRVQ):
    def __init__(self, in_dim:int=2, h_dims:list=[64,32,24,24,16,16], patch_size:tuple=(3,2), kernel_size:tuple=(5,2), depth:int=1,) -> None:
        super().__init__(backbone="conv")

        ins, outs = h_dims, h_dims[1:] + [h_dims[-1]]
    
        blocks = nn.ModuleList()
        for i in range(len(h_dims)-1):
            blocks.append(
                ConvolutionLayer(ins[i], outs[i], depth, kernel_size, transpose=True)
            )
        self.dec_blks = blocks
        self.post_conv = Convolution2D(ins[-1], outs[-1], kernel_size, scale=False)
        self.patch_deembed = PatchDeEmbed(192, in_dim, patch_size, h_dims[-1], backbone="conv")

    def forward(self, enc_hs: list, num_streams: int, quantizers: nn.ModuleList, freeze_codebook: bool=False):
        """Forward Function: Step-wise cross-scale decoding
        Args: 
            enc_hs: a list of encoded features at all scales
            num_streams: number of bitstreams to use
            quantizers: a modulelist of multi-scale quantizers
            freeze_codebook: boolean (True when no codebook is used in pretraining)
            returns: 
                recon_feat: reconstructed complex spectrum (B,2,F,T)
                dec_hs: list of decoded hidden states
                codes: discrete indices (B,num_streams,group_size,T//overlap) 
                       num_streams is always max_stream in training mode
                cm_loss, cb_loss: VQ losses (B,)
        """
        print("enc_hs[-1]: ", enc_hs[-1].shape)
        z0, cm_loss, cb_loss, code = self.csrvq(enc=enc_hs[-1], dec=0.0, vq=quantizers[0], 
                                                transmit=True, freeze_codebook=freeze_codebook)
        codes, dec_hs = [code], [z0]
        for i, blk in enumerate(self.dec_blks):
            transmit = (i < num_streams-1)    
            if self.training is True: # passing all quantizers during training
                dec_i_refine, cm_loss_i, cb_loss_i, code_i = self.csrvq(
                                                        enc=enc_hs[-1-i], dec=dec_hs[i],
                                                        vq=quantizers[i+1], transmit=transmit, freeze_codebook=freeze_codebook
                                                        )
                cm_loss += cm_loss_i
                cb_loss += cb_loss_i
                codes.append(code_i)
            else:                     # passing only transmitted quantizers during testing
                if transmit:
                    dec_i_refine, cm_loss_i, cb_loss_i, code_i = self.csrvq(
                                                        enc=enc_hs[-1-i], dec=dec_hs[i],
                                                        vq=quantizers[i+1], transmit=True, freeze_codebook=False
                                                        )
                    cm_loss += cm_loss_i
                    cb_loss += cb_loss_i
                    codes.append(code_i)
                else:
                    dec_i_refine = dec_hs[i]
            
            dec_next = blk(dec_i_refine)
            dec_hs.append(dec_next)

        recon_feat = self.post_conv(dec_next)
        recon_feat = self.patch_deembed(recon_feat)
        codes = torch.stack(codes, dim=1)   # [B, num_streams, group_size, T]
        return recon_feat, dec_hs, codes, cm_loss, cb_loss

class SwinTDecoder(nn.Module):
    def __init__(self,                  
                 in_freq: int, 
                 in_dim: int, 
                 h_dims: list,
                 patch_size: tuple,
                 swin_heads: list,
                 swin_depth: int,
                 window_size: int,
                 mlp_ratio: float,) -> None:
        super().__init__()

        self.patch_deembed = PatchDeEmbed(in_freq, in_dim, patch_size, h_dims[-1],)
        in_dims, out_dims = h_dims[:-1], h_dims[1:]
                
        self.post_swin = TransformerLayer(
                    h_dims[-1], h_dims[-1], swin_heads[-1], swin_depth, window_size, mlp_ratio, 
                    activation=nn.GELU, norm_layer=nn.LayerNorm, scale=None
                    )

        blocks = nn.ModuleList()
        for i in range(len(in_dims)):
            blocks.append(
                TransformerLayer(
                    in_dims[i], out_dims[i], swin_heads[i], swin_depth, window_size, mlp_ratio, 
                    activation=nn.GELU, norm_layer=nn.LayerNorm, scale="up", scale_factor=(2,1)
                )
            )
        self.blocks = blocks

    def forward(self, z_q, feat_shape):
        
        Wh, Ww = feat_shape

        for blk in self.blocks:
            z_q, Wh, Ww = blk(z_q, Wh, Ww)
        
        recon_x, Wh, Ww = self.post_swin(z_q, Wh, Ww)
        recon_x = self.patch_deembed(recon_x)

        return recon_x