import torch, torchaudio, math
import torch.nn as nn
from einops import rearrange
from typing import Literal

from ..modules import ProductVectorQuantize, ProductResidualVectorQuantize
from ..modules import TransformerLayer, PatchEmbed, PatchDeEmbed, Convolution2D, ConvolutionLayer
from .utils import blk_func

class BaseAudioCodec(nn.Module):
    """Base Complex STFT Audio Codec"""
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
            x (Tensor): audio tensor of shape (B, L)
        Returns: 
            complex STFT spectrum of shape (B, 2, H, W)
        """
        feat = torch.view_as_real(self.ft(x)) # B, H, W, 2
        return rearrange(feat, "b h w c -> b c h w")
        
    def audio_reconstruct(self, feat):
        """ Recover Complex STFT Spectrum to audio 
        Args: 
            feat (Tensor):    spectrum of shape (B, 2, H, W)
        Returns: 
            audio tensor of shape (B, L)
        """
        feat = torch.view_as_complex(rearrange(feat, "b c h w -> b h w c").contiguous())
        return self.ift(feat)
        
    def init_ProductVQs(self, patch_size: tuple, overlap: int, group_size: int, codebook_dims: list, codebook_size: int, l2norm: bool):
        
        freq_patch, time_patch = patch_size
        H = self.in_freq//freq_patch

        quantizers = nn.ModuleList()
        quantizers.append(
                ProductVectorQuantize( # VQ at bottom
                    in_dim=self.dec_h_dims[0], in_freq=H//2**(self.max_streams-1), 
                    overlap=overlap, num_vqs=group_size, codebook_dim=codebook_dims[0],
                    codebook_size=codebook_size, l2norm=l2norm
                )
            ) 
        for i in range(1, self.max_streams):
            quantizers.append(
                ProductVectorQuantize(
                    in_dim=self.dec_h_dims[i-1], in_freq=H//2**(self.max_streams-i),
                    overlap=overlap, num_vqs=group_size, codebook_dim=codebook_dims[i],
                    codebook_size=codebook_size, l2norm=l2norm
                )
            ) 
        self.max_bps = (2/overlap)*self.max_streams * math.log2(codebook_size)*group_size // (20*time_patch//2)
        return quantizers

    def init_ResidualVQs(self, patch_size: tuple, overlap: int, num_product_vqs: int, num_residual_vqs: int, codebook_dim: int, codebook_size: int, l2norm: bool):
        
        freq_patch, time_patch = patch_size
        H = self.in_freq//freq_patch

        quantizers = ProductResidualVectorQuantize(
            in_dim=self.dec_h_dims[0], in_freq=H//2**(self.max_streams-1),
            overlap=overlap, num_pvqs=num_product_vqs, num_rvqs=num_residual_vqs, 
            codebook_dim=codebook_dim, codebook_size=codebook_size, l2norm=l2norm
        )
        self.max_bps = (2/overlap)*self.max_streams * math.log2(codebook_size)*num_product_vqs // (20*time_patch//2)
        return quantizers

    def print_codec(self):
        if isinstance(self.quantizers, ProductResidualVectorQuantize):
            print("Codec Visualization [only at bottom]")
            print("     Freq dim:                ", self.quantizers.in_freq)
            print("     Channel(hidden) dim:     ", self.quantizers.in_dim)
            print("     Reshaped hidden dim:     ", self.quantizers.fix_dim)
            print("     Individual z_e dim:      ", self.quantizers.fix_dim*self.quantizers.overlap//len(self.quantizers.vqs))
            print("     Codebook dim:            ", self.quantizers.codebook_dim)
        else:
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


class Encoder(nn.Module):
    def __init__(self, 
                 backbone: Literal['transformer', 'convolution'] = 'transformer',
                 in_freq: int = 192, 
                 in_dim: int = 2, 
                 h_dims: list = [45,72,96,144,192,384],
                 patch_size: list = [3,2],
                 kernel_size: list=[5,2],
                 conv_depth: int=1,
                 swin_heads: list = [3,6,12,24,24],
                 swin_depth: int = 2,
                 window_size: int = 4,
                 mlp_ratio: float = 4.0,
                 ) -> None:
        super().__init__()
        
        in_dims, out_dims = h_dims[:-1], h_dims[1:]
        blocks = nn.ModuleList()
        for i in range(len(in_dims)):
            layer = ConvolutionLayer(in_dims[i], out_dims[i], conv_depth, kernel_size, transpose=False) if backbone == "convolution" \
               else TransformerLayer(
                    in_dims[i], out_dims[i], swin_heads[i], swin_depth, window_size, mlp_ratio, 
                    activation=nn.GELU, norm_layer=nn.LayerNorm, scale="down", scale_factor=(2,1)
                    )
            blocks.append(layer)

        self.patch_embed = PatchEmbed(in_freq, in_dim, patch_size, embed_dim=h_dims[0], backbone=backbone)
        self.pre_nn = Convolution2D(h_dims[0], h_dims[0], kernel_size, scale=False) if backbone == "convolution" \
                 else TransformerLayer(h_dims[0], h_dims[0], swin_heads[0], swin_depth, window_size, mlp_ratio,
                                         activation=nn.GELU, norm_layer=nn.LayerNorm, scale=None)
        self.blocks = blocks
        self.patch_size = patch_size

    def forward(self, x):
        """Forward Function: step-wise encoding with downscaling
        Args: 
            x: complex spectrum feature, tensor with shape (B, C=2, F, T)
            returns: encoder hidden states at all scales; patch num at bottom
        """
        feat_shape = (x.size(2)//self.patch_size[0], x.size(3)//self.patch_size[1])
        x = self.patch_embed(x) 

        x, feat_shape = blk_func(self.pre_nn, x, feat_shape)
        enc_hs = [x]
        for blk in self.blocks:
            x, feat_shape = blk_func(blk, x, feat_shape)
            enc_hs.append(x)

        return enc_hs, feat_shape
    

class Decoder(nn.Module):
    def __init__(self,        
                 backbone: Literal['transformer', 'convolution'] = 'transformer',          
                 in_freq: int = 192, 
                 in_dim: int = 2, 
                 h_dims: list = [384,192,144,96,72,45],
                 patch_size: list = [3,2],
                 kernel_size: list=[5,2],
                 conv_depth: int=1,
                 swin_heads: list = [24,24,12,6,3],
                 swin_depth: int = 2,
                 window_size: int = 4,
                 mlp_ratio: float = 4.0,) -> None:
        super().__init__()

        in_dims, out_dims = h_dims[:-1], h_dims[1:]

        blocks = nn.ModuleList()
        for i in range(len(in_dims)):
            layer = ConvolutionLayer(in_dims[i], out_dims[i], conv_depth, kernel_size, transpose=True) if backbone == "convolution" \
               else TransformerLayer(
                    in_dims[i], out_dims[i], swin_heads[i], swin_depth, window_size, mlp_ratio, 
                    activation=nn.GELU, norm_layer=nn.LayerNorm, scale="up", scale_factor=(2,1)
                )
            blocks.append(layer)
            
        self.patch_deembed = PatchDeEmbed(in_freq, in_dim, patch_size, h_dims[-1], backbone)
        self.post_nn = Convolution2D(h_dims[-1], h_dims[-1], kernel_size, scale=False) if backbone == "convolution" \
                  else TransformerLayer(
                       h_dims[-1], h_dims[-1], swin_heads[-1], swin_depth, window_size, mlp_ratio, 
                       activation=nn.GELU, norm_layer=nn.LayerNorm, scale=None
                    )
        self.blocks = blocks
    
    def forward(self, z_q, feat_shape):
        
        for blk in self.blocks:
            z_q, feat_shape = blk_func(blk, z_q, feat_shape)
        
        recon_feat, feat_shape = blk_func(self.post_nn, z_q, feat_shape)
        recon_feat = self.patch_deembed(recon_feat)

        return recon_feat