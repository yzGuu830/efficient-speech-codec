import torch, torchaudio, math
import torch.nn as nn
from einops import rearrange

from modules import TransformerLayer, PatchEmbed, PatchDeEmbed, ProductVectorQuantize

class BaseAudioCodec(nn.Module):
    def __init__(self, in_dim: int, in_freq: int, h_dims: list, max_streams: int,
                 win_len: int=20, hop_len: int=5, sr: int=16000) -> None:
        super().__init__()

        self.in_freq, self.in_dim = in_freq, in_dim
        self.max_streams = max_streams

        self.enc_h_dims = h_dims
        self.dec_h_dims = h_dims[::-1]

        self.ft = torchaudio.transforms.Spectrogram(n_fft=(in_freq-1)*2, 
                                            win_length=int(win_len*sr*1e-3),
                                            hop_length=int(hop_len*sr*1e-3), power=None)
        self.ift = torchaudio.transforms.InverseSpectrogram(n_fft=(in_freq-1)*2, 
                                            win_length=int(win_len*sr*1e-3),
                                            hop_length=int(hop_len*sr*1e-3))
        
    def spec_transform(self, x):
        """ Transform audio to Complex Spectrum
        Args: 
            x: audio tensor of shape [B, L]
            returns: spectrum of shape [B, 2, H, W]
        """
        feat = torch.view_as_real(self.ft(x)) # B, H, W, 2
        return rearrange(feat, "b h w c -> b c h w")
        
    def audio_reconstruct(self, feat):
        """ Recover Complex Spectrum to audio 
        Args: 
            feat:    spectrum of shape [B, 2, H, W]
            returns: audio tensor of shape [B, L]
        """
        feat = torch.view_as_complex(rearrange(feat, "b c h w -> b h w c"))
        return self.ift(feat)
        
    def init_quantizers(self, patch_size: tuple, overlap: int, group_size: int, codebook_dims: list, 
                codebook_size: int, l2norm: bool, init_method: str):
        
        freq_patch, time_patch = patch_size
        H = self.in_freq//freq_patch

        if init_method == "kaiming": 
            kmeans_init = None
        elif init_method == "kmeans": # requires pretraining
            kmeans_init = True
        elif init_method == "randomfill": # requires pretraining
            kmeans_init = False
        else:
            raise ValueError("\{kmeans_init\} should be in (kaiming, kmeans, randomfill)")

        quantizers = nn.ModuleList()
        quantizers.append(
                ProductVectorQuantize( # VQ at bottom
                    in_dim=self.dec_h_dims[0], in_freq=H//2**(self.max_streams-1), 
                    overlap=overlap, num_vqs=group_size, codebook_dim=codebook_dims[0],
                    codebook_size=codebook_size, l2norm=l2norm, kmeans_init=kmeans_init
                )
            ) 
        for i in range(1, self.max_streams):
            quantizers.append(
                ProductVectorQuantize(
                    in_dim=self.dec_h_dims[i-1], in_freq=H//2**(self.max_streams-i),
                    overlap=overlap, num_vqs=group_size, codebook_dim=codebook_dims[i],
                    codebook_size=codebook_size, l2norm=l2norm, kmeans_init=kmeans_init
                )
            ) 
        
        self.max_bps = (2/overlap)*self.max_streams * math.log2(codebook_size)*group_size // (20*time_patch//2)
        return quantizers

    def print_codec(self):
        freq_dims, hidden_dims, reshaped_dims, individual_dims, codebook_dims = [], [], [], [], []
        for pvq in self.quantizers:
            freq_dims.append(pvq.in_freq)
            hidden_dims.append(pvq.in_dim)
            reshaped_dims.append(pvq.fix_dim)
            individual_dims.append(pvq.fix_dim*pvq.overlap//pvq.num_vqs)
            codebook_dims.append(pvq.codebook_dim)
            
        print("Codec Visualization [from bottom to top]: ")
        print("     Freq dims:                ", freq_dims)
        print("     Channel(hidden) dims:     ", hidden_dims)
        print("     Reshaped hidden dims:     ", reshaped_dims)
        print("     Individual z_e dims:      ", individual_dims)
        print("     Codebook dims:            ", codebook_dims)
    
class ESC(BaseAudioCodec):
    def __init__(self, in_dim: int=2, in_freq: int=192, h_dims: list=[45,45,72,96,192,384],
                 max_streams: int=6, win_len: int=20, hop_len: int=5, sr: int=16000,
                 patch_size: list = [3,2], swin_heads: list = [3,3,6,12,24], swin_depth: int = 2,
                 window_size: int = 4, mlp_ratio: float = 4.,
                 overlap: int = 2, group_size: int = 6, 
                 codebook_size: int = 1024, codebook_dims: list = [8,8,8,8,8,8], 
                 l2norm: bool = True, init_method: str = "kmeans",) -> None:
        super().__init__(in_dim, in_freq, h_dims, max_streams, win_len, hop_len, sr)

        self.quantizers = self.init_quantizers(
                    patch_size, overlap, group_size, codebook_dims, codebook_size, l2norm, init_method
                )
        self.encoder = Encoder(in_freq, in_dim, h_dims, tuple(patch_size), swin_heads, swin_depth, window_size, mlp_ratio)
        self.decoder = Decoder(in_freq, in_dim, h_dims[::-1], tuple(patch_size), swin_heads[::-1], swin_depth, window_size, mlp_ratio)

    def forward_one_step(self, x, x_feat=None, num_streams=6, freeze_codebook=False):
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = rearrange(x_feat, "b h w c -> b c h w") 

        enc_hs, H,W = self.encoder(x_feat)
        recon_feat, _, codes, cm_loss, cb_loss = self.decoder(enc_hs, num_streams, self.quantizers, (H,W), freeze_codebook)
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
        
    @torch.no_grad()
    def encode(self, x, num_streams=6):
        x_feat = self.spec_transform(x)
        enc_hs, H, W = self.encoder(x_feat)
        codes = self.decoder.encode(enc_hs, num_streams, self.quantizers, (H,W))
        return codes, (H,W)
    
    @torch.no_grad()
    def decode(self, codes, feat_shape=(2,300)):
        dec_hs = self.decoder.decode(codes, self.quantizers, feat_shape)
        recon_feat = dec_hs[-1]
        recon_x = self.audio_reconstruct(recon_feat)
        return recon_x
    
    def vis_encoder_decoder(self):
        self.encoder.vis_encoder()
        print()
        self.decoder.vis_decoder()

class Encoder(nn.Module):
    def __init__(self, 
                 in_freq: int, 
                 in_dim: int, 
                 h_dims: list,
                 patch_size: tuple,
                 swin_heads: list,
                 swin_depth: int,
                 window_size: int,
                 mlp_ratio: float,
                 ) -> None:
        super().__init__()
        
        self.patch_embed = PatchEmbed(in_freq, in_dim, patch_size, embed_dim=h_dims[0])
        in_dims, out_dims = h_dims[:-1], h_dims[1:]

        self.pre_swin = TransformerLayer(h_dims[0], h_dims[0], swin_heads[0], swin_depth, window_size, mlp_ratio,
                                         activation=nn.GELU, norm_layer=nn.LayerNorm, scale=None)
        self.patch_size = patch_size

        blocks = nn.ModuleList()
        for i in range(len(in_dims)):
            blocks.append(
                TransformerLayer(
                    in_dims[i], out_dims[i], swin_heads[i], swin_depth, window_size, mlp_ratio, 
                    activation=nn.GELU, norm_layer=nn.LayerNorm, scale="down", scale_factor=(2,1)
                    )
            )
        self.blocks = blocks

    def forward(self, x):
        """Forward Function: Step-wise encoding with downscaling
        Args: 
            x: complex spectrum feature, tensor with shape (B, C=2, F, T)
            returns: encoder hidden states at all scales; patch num at bottom
        """
        Wh, Ww = x.size(2)//self.patch_size[0], x.size(3)//self.patch_size[1]
        x = self.patch_embed(x)                 # B C Wh Ww
        x, Wh, Ww = self.pre_swin(x, Wh, Ww)    # B C Wh Ww

        enc_hs = [x]
        for blk in self.blocks:
            x, Wh, Ww = blk(x, Wh, Ww)
            enc_hs.append(x)

        return enc_hs, Wh, Ww
    
    def vis_encoder(self):
        blk = self.pre_swin
        print("Pre-swin Layer: swin_depth={} swin_hidden={} heads={} down={}".format(
                blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))
        
        for i, blk in enumerate(self.blocks):
            print("Layer[{}]: swin_depth={} swin_hidden={} heads={} down={}".format(
                i, blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))
    
class CrossScaleDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def pre_fuse(self, enc, dec):
        """Compute residuals to quantize"""
        return enc - dec
    
    def post_fuse(self, residual_q, dec):
        """Add back quantized residuals"""
        return residual_q + dec

    def csrvq(self, enc: torch.tensor, dec: torch.tensor, vq: nn.Module, 
            transmit: bool=True, freeze_codebook: bool=False):
        """ Forward Function.
        Args:
            enc: encoded feature tensor with shape (B, H*W, C)
            dec: decoded feature tensor with shape (B, H*W, C)
            vq: product quantizer at this stream level
            transmit: whether this stream is transmitted (perform quantization or not)
            freeze_codebook: whether freeze the codebook (in a pre-training stage)
            returns: dec_refine (decoded feature conditioned on quantized encodings)
        """

        residual = self.pre_fuse(enc, dec)
        outputs, losses = vq(residual, freeze_codebook)
        residual_q, _, code = outputs
        cm_loss, cb_loss = losses

        if not transmit: # stop updates at non-transmitted streams
            cm_loss, cb_loss = cm_loss*0., cb_loss*0.
            residual_q *= 0.

        dec_refine = self.post_fuse(residual_q, dec)
        return dec_refine, cm_loss, cb_loss, code
    
    def csrvq_encode(self, enc, dec, vq):

        residual = self.pre_fuse(enc, dec)
        code = vq.encode(residual)
        return code
    
    def csrvq_decode(self, codes, dec, vq):

        residual_q = vq.decode(codes)
        dec_refine = self.post_fuse(residual_q, dec)
        return dec_refine
    
class Decoder(CrossScaleDecoder):
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

    def forward(self, enc_hs: list, num_streams: int, quantizers: nn.ModuleList, 
                feat_shape: tuple, freeze_codebook: bool=False):
        """Forward Function: Step-wise cross-scale decoding
        Args: 
            enc_hs: a list of encoded features at all scales
            num_streams: number of bitstreams to use
            quantizers: a modulelist of multi-scale quantizers
            feat_shape: (Wh, Ww) feature shape at bottom level
            freeze_codebook: boolean (True when no codebook is used in pretraining)
            returns: 
                recon_feat: reconstructed complex spectrum (B,2,F,T)
                dec_hs: list of decoded hidden states
                codes: discrete indices (B,num_streams,group_size,T//overlap) 
                       num_streams is always max_stream in training mode
                cm_loss, cb_loss: VQ losses (B,)
        """
        Wh, Ww = feat_shape
        z0, cm_loss, cb_loss, code = self.csrvq(enc=enc_hs[-1], dec=0.0, vq=quantizers[0], 
                                                transmit=True, freeze_codebook=freeze_codebook)
        codes, dec_hs = [code], [z0]
        for i, blk in enumerate(self.blocks):
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
            
            dec_next, Wh, Ww = blk(dec_i_refine, Wh, Ww)
            dec_hs.append(dec_next)

        dec_next, Wh, Ww = self.post_swin(dec_next, Wh, Ww)
        recon_feat = self.patch_deembed(dec_next)
        codes = torch.stack(codes, dim=1) # [B, num_streams, group_size, T]
        return recon_feat, dec_hs, codes, cm_loss, cb_loss
    
    def encode(self, enc_hs: list, num_streams: int, quantizers: nn.ModuleList, feat_shape: tuple):
        """Encode audio into indices
        Args: 
            enc_hs: a list of encoded features at all scales
            num_streams: number of bitstreams to use
            quantizers: a modulelist of multi-scale quantizers
            feat_shape: (Wh, Ww) feature shape at bottom level
        returns: multi-scale codes with shape (B, num_streams, group_size, T)
        """
        Wh, Ww = feat_shape
        code0 = quantizers[0].encode(enc_hs[-1]) # [B, group_size, T]
        if num_streams == 1:
            return code0.unsqueeze(1)
        
        z0 = quantizers[0].decode(code0)
        codes, dec_hs = [code0], [z0]
        for i in range(num_streams-1):
            
            codei = self.csrvq_encode(enc=enc_hs[-1-i], dec=dec_hs[i], vq=quantizers[i+1])
            codes.append(codei)
            if len(codes) == num_streams: break

            dec_i_refine = self.csrvq_decode(codei, dec=dec_hs[i], vq=quantizers[i+1])
            dec_next, Wh, Ww = self.blocks[i](dec_i_refine, Wh, Ww)
            dec_hs.append(dec_next)
        
        codes = torch.stack(codes, dim=1) 
        return codes    # [B, num_streams, group_size, T]
    
    def decode(self, codes: list, quantizers: nn.ModuleList, feat_shape: tuple):
        """Decode from indices
        Args: 
            codes: multi-scale codes with shape (B, num_streams, group_size, T)
            quantizers: a modulelist of multi-scale quantizers
            feat_shape: (Wh, Ww) feature shape at bottom level
        returns: decoded hidden states
        """
        Wh, Ww = feat_shape
        num_streams = codes.size(1)

        z0 = quantizers[0].decode(codes[:, 0])
        dec_hs = [z0]
        for i in range(len(self.blocks)): # using code of residuals to refine decoding
            if i < num_streams-1:
                dec_i_refine = self.csrvq_decode(codes=codes[:, i+1], dec=dec_hs[i], vq=quantizers[i+1])
            else:
                dec_i_refine = dec_hs[i]
            dec_next, Wh, Ww = self.blocks[i](dec_i_refine, Wh, Ww)
            dec_hs.append(dec_next)

        dec_next, Wh, Ww = self.post_swin(dec_next, Wh, Ww)
        dec_hs.append(self.patch_deembed(dec_next))
        return dec_hs

    def vis_decoder(self):
        
        for i, blk in enumerate(self.blocks):
            print("Layer[{}]: swin_depth={} swin_hidden={} heads={} up={}".format(
                i, blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))

        blk = self.post_swin
        print("Post-swin Layer: swin_depth={} swin_hidden={} heads={} up={}".format(
                blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))

def make_model(model_config):
    if isinstance(model_config, dict):
        model = ESC(**model_config)
    else:
        model = ESC(**vars(model_config))
    return model