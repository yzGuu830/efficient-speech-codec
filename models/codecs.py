import torch
import torch.nn as nn
from einops import rearrange

from models.esc import BaseAudioCodec, CrossScaleRVQ, SwinTEncoder
from modules import ConvolutionLayer, TransformerLayer, Convolution2D, PatchDeEmbed

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
            self.encoder = ConvEncoder(in_dim, h_dims, tuple(kernel_size), conv_depth)
            self.decoder = ConvDecoder(in_dim, h_dims[::-1], tuple(kernel_size), conv_depth) 
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
    def __init__(self, in_dim: int, in_freq: int, h_dims: list, max_streams: int, win_len: int = 20, hop_len: int = 5, sr: int = 16000) -> None:
        super().__init__(in_dim, in_freq, h_dims, max_streams, win_len, hop_len, sr)

        self.quantizers = self.init_product_vqs()
        self.encoder = ConvEncoder
        self.decoder = CrossScaleConvDecoder


class ConvEncoder(nn.Module):
    def __init__(self, in_dim:int=2, h_dims:list=[16,16,24,24,32,64], kernel_size:tuple=(5,2), depth:int=1) -> None:
        super().__init__()

        ins, outs = [in_dim] + h_dims, h_dims
        self.pre_conv = Convolution2D(ins[0], outs[0], kernel_size, scale=False)

        blocks = nn.ModuleList()
        for i in range(1, len(h_dims)):
            blocks.append(
                ConvolutionLayer(ins[i], outs[i], depth, kernel_size,)
            )
        self.enc_blks = blocks

    def forward(self, x):
        x = self.pre_conv(x)

        enc_hs = [x]
        for blk in self.enc_blks:
            x = blk(x)
            enc_hs.append(x)
        
        return enc_hs, x.size(2), x.size(3)


class ConvDecoder(nn.Module):
    def __init__(self, in_dim:int=2, h_dims:list=[64,32,24,24,16,16], kernel_size:tuple=(5,2), depth:int=1) -> None:
        super().__init__()
        ins, outs = h_dims, h_dims[1:] + [in_dim]
    
        blocks = nn.ModuleList()
        for i in range(len(h_dims)-1):
            blocks.append(
                ConvolutionLayer(ins[i], outs[i], depth, kernel_size, transpose=True)
            )
        self.dec_blks = blocks
        self.post_conv = Convolution2D(ins[-1], outs[-1], kernel_size, scale=False)

    def forward(self, z_q, feat_shape):
    
        for blk in self.dec_blks:
            z_q = blk(z_q)

        recon_x = self.post_conv(z_q)
        return recon_x


class CrossScaleConvDecoder(CrossScaleRVQ):
    def __init__(self, backbone="conv") -> None:
        super().__init__(backbone)


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