import torch
from einops import rearrange
from typing import Literal

from .base import BaseAudioCodec, Encoder, Decoder
from .csrvq import CrossScaleRVQDecoder


class ESC(BaseAudioCodec):
    """ Efficient Speech Codec """
    def __init__(self, in_dim: int=2, in_freq: int=192, h_dims: list=[45,72,96,144,192,384],
                 max_streams: int=6, win_len: int=20, hop_len: int=5, sr: int=16000,
                 patch_size: list = [3,2], swin_heads: list = [3,6,12,24,24], swin_depth: int = 2,
                 window_size: int = 4, mlp_ratio: float = 4.,
                 overlap: int = 2, group_size: int = 3, 
                 codebook_size: int = 1024, codebook_dims: list = [8,8,8,8,8,8], 
                 l2norm: bool = True, backbone: Literal['transformer', 'convolution'] = 'transformer',
                 kernel_size: list=[5,2], conv_depth: int=1) -> None:
        super().__init__(in_dim, in_freq, h_dims, max_streams, win_len, hop_len, sr)

        self.quantizers = self.init_ProductVQs(patch_size, overlap, group_size, codebook_dims, codebook_size, l2norm)
        self.encoder = Encoder(backbone, in_freq, in_dim, h_dims, tuple(patch_size), tuple(kernel_size), conv_depth, 
                               swin_heads, swin_depth, window_size, mlp_ratio)
        self.decoder = CrossScaleRVQDecoder(backbone=backbone, in_freq=in_freq, in_dim=in_dim, 
                                            h_dims=h_dims[::-1], patch_size=tuple(patch_size), 
                                            swin_heads=swin_heads[::-1], swin_depth=swin_depth, 
                                            window_size=window_size, mlp_ratio=mlp_ratio,
                                            kernel_size=tuple(kernel_size), conv_depth=conv_depth,)

    def forward_one_step(self, x, x_feat=None, num_streams=6, freeze_codebook=False):
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = rearrange(x_feat, "b h w c -> b c h w") 

        enc_hs, feat_shape = self.encoder(x_feat)
        recon_feat, codes, cm_loss, cb_loss = self.decoder(enc_hs, num_streams, self.quantizers, feat_shape, freeze_codebook)
        recon_x = self.audio_reconstruct(recon_feat)

        return {"cm_loss": cm_loss,
                "cb_loss": cb_loss,
                "raw_audio": x,
                "recon_audio": recon_x,
                "raw_feat": x_feat, 
                "recon_feat": recon_feat,
                "codes": codes}
    
    def forward(self, x, x_feat, num_streams, freeze_codebook=False):
        """ Forward Function.
        Args: 
            x (Tensor): audio waveform with shape (Bs, L)
            x_feat (Tensor): audio complex STFT with shape (Bs, 2, F, T)
            num_streams (int): number of streams transmitted
            freeze_codebook (boolean): set True during pre-training stage
        Returns:
            dict:
                cm_loss: commitment loss with shape (Bs, )
                cb_loss: codebook loss with shape (Bs, )
                raw_audio: audio waveform with shape (Bs, L)
                recon_audio: reconstructed audio waveform with shape (Bs, L)
                raw_feat: audio complex STFT with shape (Bs, 2, F, T)
                recon_feat: reconstructed audio complex STFT with shape (Bs, 2, F, T)
                codes: multi-scale indices with shape (Bs, num_streams, group_size, *)
        """
        num_streams = self.max_streams if freeze_codebook else num_streams
        return self.forward_one_step(x, x_feat, num_streams, freeze_codebook)
        
    @torch.no_grad()
    def encode(self, x, num_streams=6):
        """ Encoding.
        Args:
            x (Tensor): audio waveform with shape (Bs, L)
            num_streams (int): number of streams transmitted
        Returns:
            Tensor of multi-scale codes with shape (Bs, num_streams, group_size, *) 
            Tuple of latent feature shape (H,W)
        """
        x_feat = self.spec_transform(x)
        enc_hs, feat_shape = self.encoder(x_feat)
        codes = self.decoder.encode(enc_hs, num_streams, self.quantizers, feat_shape)
        return codes, feat_shape
    
    @torch.no_grad()
    def decode(self, codes, feat_shape=(2,1000)):
        """Decoding.
        Args:
            codes (Tensor): multi-scale codes with shape (Bs, num_streams, group_size, *)
            feat_shape (tuple): latent feature shape (H, W)
        Returns:
            Tensor of reconstructed audio waveform
        """
        dec_hs = self.decoder.decode(codes, self.quantizers, feat_shape)
        recon_x = self.audio_reconstruct(dec_hs[-1])
        return recon_x

class RVQCodecs(BaseAudioCodec):
    """ RVQ Codecs """
    def __init__(self, 
                in_dim: int=2, 
                in_freq: int=192, 
                h_dims: list=[45,72,96,144,192,384], # [16,16,24,24,32,64]
                max_streams: int=6,   # max layer depth here
                backbone: Literal['transformer', 'convolution'] = 'transformer',
                kernel_size: list=[5,2],
                conv_depth: int=1,
                patch_size: list=[3,2],
                swin_heads: list=[3,6,12,24,24], 
                swin_depth: int=2, 
                window_size: int=4, 
                mlp_ratio: float=4.,
                overlap: int=2,
                num_rvqs: int=6,
                group_size: int=3,
                codebook_dim: int=8,
                codebook_size: int=1024,
                l2norm: bool=True,
                win_len: int = 20, 
                hop_len: int = 5, 
                sr: int = 16000) -> None:
        super().__init__(in_dim, in_freq, h_dims, max_streams, win_len, hop_len, sr)

        self.quantizers = self.init_ResidualVQs(patch_size, overlap, group_size, num_rvqs, codebook_dim, codebook_size, l2norm)
        self.encoder = Encoder(backbone, in_freq, in_dim, h_dims, patch_size, kernel_size, conv_depth, swin_heads, swin_depth, window_size, mlp_ratio)
        self.decoder = Decoder(backbone, in_freq, in_dim, h_dims[::-1], patch_size, kernel_size, conv_depth, swin_heads[::-1], swin_depth, window_size, mlp_ratio)
        self.dims = 3 if backbone == "transformer" else 4

    def forward_one_step(self, x, x_feat=None, num_streams=6, freeze_codebook=False):
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = rearrange(x_feat, "b h w c -> b c h w") 

        enc_hs, feat_shape = self.encoder(x_feat)

        outputs = self.quantizers(enc_hs[-1], num_streams, freeze_vq=freeze_codebook)
        cm_loss, cb_loss = outputs['cm_loss'], outputs['cb_loss']
        z_q, codes = outputs['z_q'], outputs['codes']

        recon_feat = self.decoder(z_q, feat_shape)
        recon_x = self.audio_reconstruct(recon_feat)

        return {"cm_loss": cm_loss,
                "cb_loss": cb_loss,
                "raw_audio": x,
                "recon_audio": recon_x,
                "raw_feat": x_feat, 
                "recon_feat": recon_feat,
                "codes": codes}

    def forward(self, x, x_feat, num_streams, freeze_codebook=False):
        """ Forward Function.
        Args: 
            x (Tensor): audio waveform with shape (Bs, L)
            x_feat (Tensor): audio complex STFT with shape (Bs, 2, F, T)
            num_streams (int): number of streams transmitted
            freeze_codebook (boolean): set True during pre-training stage
        Returns:
            dict:
                cm_loss: commitment loss with shape (Bs, )
                cb_loss: codebook loss with shape (Bs, )
                raw_audio: audio waveform with shape (Bs, L)
                recon_audio: reconstructed audio waveform with shape (Bs, L)
                raw_feat: audio complex STFT with shape (Bs, 2, F, T)
                recon_feat: reconstructed audio complex STFT with shape (Bs, 2, F, T)
                codes: multi-scale indices with shape (Bs, num_streams, group_size, *)
        """
        return self.forward_one_step(x, x_feat, num_streams, freeze_codebook)

    @torch.no_grad()
    def encode(self, x, num_streams=6):
        x_feat = self.spec_transform(x)
        enc_hs, feat_shape = self.encoder(x_feat)
        codes = self.quantizers.encode(enc_hs[-1], num_streams)
        return codes, feat_shape
    
    @torch.no_grad()
    def decode(self, codes, feat_shape):
        z_q = self.quantizers.decode(codes, dims=self.dims)
        recon_feat = self.decoder(z_q, feat_shape)
        recon_x = self.audio_reconstruct(recon_feat)
        return recon_x

model_dict = {
    "csvq+conv": ESC,
    "csvq+swinT": ESC,
    "rvq+conv": RVQCodecs,
    "rvq+swinT": RVQCodecs
}

def make_model(model_config, model_name):
    if model_name not in model_dict:
        assert f'{model_name} is not valid within [csvq+conv, csvq+swinT, rvq+conv, rvq+swinT]'
    
    m = model_dict[model_name]
    if isinstance(model_config, dict):
        model = m(**model_config)
    else:
        model = m(**vars(model_config))
        
    return model