import sys
sys.path.append("/Users/tracy/Library/CloudStorage/GoogleDrive-cloudstorage.yuzhe@gmail.com/My Drive/Research/Audio_Signal_Coding/Deep-Audio-Signal-Coding/src")

from models.convolution import ConvEncoderLayer, ConvDecoderLayer, Convolution2D
from models.tfs import TCM, RNNFilter
from models.swin import SwinTLayer, PatchEmbed, PatchDeEmbed, PatchMerging, PatchSplit, WindowAlignment
from models.vector_quantization import GroupVQ
from models.losses import MSELoss, MELLoss, TimeLoss, FreqLoss

import torch, torchaudio, math
import torch.nn as nn
import numpy as np


class BaseCodec(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 in_freq: int,
                 h_dims: list,
                 max_streams: int,
                 proj: int,
                 overlap: int,
                 num_vqs: int,
                 codebook_size: int,
                 mel_nfft: int, 
                 mel_bins: int,
                 vq_commit: float,
                 fuse_net: bool,
                 scalable: bool,
                 spec_augment: bool,
                 win_len: int, 
                 hop_len: int, 
                 sr: int,
                 ) -> None:
        super().__init__()

        self.in_freq, self.in_dim = in_freq, in_dim
        self.max_streams = max_streams

        self.enc_h_dims = h_dims
        self.dec_h_dims = h_dims[::-1]

        self.scalable = scalable
        self.fuse_net = fuse_net
        self.spec_augment = spec_augment

        self.ft = torchaudio.transforms.Spectrogram(n_fft=(in_freq-1)*2, 
                                            win_length=int(win_len*sr*1e-3),
                                            hop_length=int(hop_len*sr*1e-3), power=None)
        self.ift = torchaudio.transforms.InverseSpectrogram(n_fft=(in_freq-1)*2, 
                                            win_length=int(win_len*sr*1e-3),
                                            hop_length=int(hop_len*sr*1e-3))

        self.recon_loss = MSELoss()
        self.mel_loss = MELLoss(mel_nfft, mel_bins)

        # self.recon_loss = TimeLoss()
        # self.mel_loss = FreqLoss(n_mels=64)
        
    def init_quantizer(self, proj, overlap, num_vqs, codebook_size, vq_commit, patch_size=None, cosine_similarity=False):
        quantizer = nn.ModuleList()
        if len(proj) == 1:
            proj = proj[0]
            proj = [proj for _ in range(self.max_streams)]

        if patch_size:
            freq_patch, time_patch = patch_size[0], patch_size[1]
        else:
            freq_patch, time_patch = 1, 1
        quantizer.append(
                GroupVQ(
                    self.dec_h_dims[0], (self.in_freq // freq_patch) // 2**(len(self.enc_h_dims)-1), 
                    proj[0], overlap, num_vqs, codebook_size, vq_commit, cosine_similarity,
                )
            )
        for i in range(1, self.max_streams):
            quantizer.append(
                GroupVQ(
                    self.dec_h_dims[i-1], (self.in_freq // freq_patch) // 2**(len(self.enc_h_dims)-i),
                    proj[i], overlap, num_vqs, codebook_size, vq_commit, cosine_similarity,
                )
            ) 
        self.max_bps = self.max_streams * math.log2(codebook_size) * num_vqs // 20
        # print(f"Audio Codec {self.max_bps}kbps Initialized")
        return quantizer
    
    def vis_quantization(self):
        Hs, in_dims, merge_dims = [], [], []
        down_maps = []
        for vq in self.quantizer:
            Hs.append(vq.H)
            in_dims.append(vq.in_dim)
            merge_dims.append(vq.H*vq.in_dim)
            down_maps.append(vq.fix_dim)
        print("Quantization Vis: ")
        print("     Freq dims: ", Hs)
        print("     Channel(hidden) dims: ", in_dims)
        print("     projections from: ", merge_dims)
        print("     projections to: ", down_maps)
        print("     group_vq_dims: ", [d*vq.overlap for d in down_maps])
    
    def spec_transform(self, x):
        """
        Args: x is audio of shape [B, L]
            returns: spectrogram with shape [B, C, H, W]
        """

        feat = torch.view_as_real(self.ft(x)) # B, H, W, C
        return feat.permute(0,3,1,2)
        
    def audio_reconstruct(self, feat):
        """
        Args: feat is spectrogram of shape [B, C, H, W]
            returns: audio with shape [B, L]
        """

        feat = torch.view_as_complex(feat.permute(0,2,3,1).contiguous()) # B, H, W
        return self.ift(feat)

class SwinCrossScaleCodec(BaseCodec):
    def __init__(self, 
                 patch_size: list = [3,2],
                 swin_depth: int = 2,
                 swin_heads: int = 2,
                 window_size: int = 4,
                 mlp_ratio: float = 4.,
                 in_dim: int = 2, 
                 in_freq: int = 192, 
                 h_dims: list = [16,16,32,32,64,128], 
                 max_streams: int = 6, 
                 proj: int = 8, 
                 overlap: int = 4, 
                 num_vqs: int = 6, 
                 codebook_size: int = 1024, 
                 cosine_similarity: bool = False,
                 mel_nfft: int = 2048, 
                 mel_bins: int = 64, 
                 vq_commit: float = 1., 
                 fuse_net: bool = False, 
                 scalable: bool = False, 
                 spec_augment: bool = False, 
                 win_len: int = 20, 
                 hop_len: int = 5, 
                 sr: int = 16000,
                 vis: bool = True) -> None:
        super().__init__(in_dim, in_freq, h_dims, max_streams, proj, overlap, num_vqs, codebook_size, mel_nfft, mel_bins, vq_commit, fuse_net, scalable, spec_augment, win_len, hop_len, sr)

        self.encoder = SwinEncoder(
            swin_depth, swin_heads, window_size, mlp_ratio,
            in_freq, patch_size, self.in_dim, self.enc_h_dims
        )
        self.decoder = SwinCrossScaleDecoder(
            swin_depth, swin_heads, window_size, mlp_ratio,
            in_freq, patch_size, self.in_dim, self.dec_h_dims,
            max_streams, fuse_net
        )
        self.quantizer = self.init_quantizer(proj, overlap, num_vqs, codebook_size, vq_commit, patch_size, cosine_similarity)

        if vis:
            self.vis_quantization()

    def train_one_step(self, x, x_feat, streams):
        self.train()
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = x_feat.permute(0,3,1,2)

        enc_hs, Wh, Ww = self.encoder.encode(x_feat)
        if self.scalable:
            streams = np.random.randint(1, self.max_streams+1)

        dec_hs, vq_loss = self.decoder.decode(enc_hs, streams, vqs=self.quantizer, Wh=Wh, Ww=Ww)

        rec_feat = dec_hs[-1]
        rec_x = self.audio_reconstruct(rec_feat)

        # recon_loss = self.recon_loss(x_feat, rec_feat)
        recon_loss = self.recon_loss(x, rec_x)
        mel_loss = self.mel_loss(x, rec_x)

        return {
                "loss": recon_loss + 0.25 * vq_loss + 0.25 * mel_loss,
                "recon_loss": recon_loss,
                "vq_loss": vq_loss,
                "mel_loss": mel_loss,
                "raw_audio": x,
                "recon_audio": rec_x,
                "raw_feat": x_feat, 
                "recon_feat": rec_feat,
            }
    @torch.inference_mode()
    def test_one_step(self, x, x_feat, streams):
        self.eval()
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = x_feat.permute(0,3,1,2)

        enc_hs, Wh, Ww = self.encoder.encode(x_feat)

        dec_hs, _ = self.decoder.decode(enc_hs, streams, vqs=self.quantizer, Wh=Wh, Ww=Ww)

        rec_feat = dec_hs[-1]
        rec_x = self.audio_reconstruct(rec_feat)

        return {
                "raw_audio": x,
                "recon_audio": rec_x,
                "raw_feat": x_feat, 
                "recon_feat": rec_feat,
            }
    
    def forward(self, x, x_feat, streams, train=False):
        if train:
            return self.train_one_step(x, x_feat, streams)
        else:
            return self.test_one_step(x, x_feat, streams)

class ConvCrossScaleCodec(BaseCodec):
    def __init__(self, 
                 in_dim: int = 2, 
                 in_freq: int = 192, 
                 h_dims: list = [16,16,24,24,32,64], 
                 max_streams: int = 6, 
                 proj: int = 2, 
                 overlap: int = 4, 
                 num_vqs: int = 6, 
                 codebook_size: int = 1024, 
                 mel_nfft: int = 2048, 
                 mel_bins: int = 64, 
                 vq_commit: float = 1., 
                 fuse_net: bool = False, 
                 scalable: bool = False, 
                 spec_augment: bool = False, 
                 use_tf: bool = True,
                 win_len: int = 20, 
                 hop_len: int = 5, 
                 sr: int = 16000) -> None:
        super().__init__(in_dim, in_freq, h_dims, max_streams, proj, overlap, num_vqs, codebook_size, mel_nfft, mel_bins, vq_commit, fuse_net, scalable, spec_augment, win_len, hop_len, sr)

        self.encoder = ConvEncoder(
            self.in_dim, self.enc_h_dims, use_tf
        )
        self.decoder = ConvCrossScaleDecoder(
            self.in_dim, self.dec_h_dims, self.fuse_net, self.max_streams, use_tf
        )
        self.quantizer = self.init_quantizer(proj, overlap, num_vqs, codebook_size, vq_commit)
        self.vis_quantization()

    def train_one_step(self, x, x_feat, streams):
        self.train()
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = x_feat.permute(0,3,1,2)

        enc_hs, enc_out = self.encoder.encode(x_feat)

        if self.scalable:
            streams = np.random.randint(1, self.max_streams+1)

        dec_hs, vq_loss = self.decoder.decode(enc_hs, enc_out, streams, vqs=self.quantizer)

        rec_feat = dec_hs[-1]
        rec_x = self.audio_reconstruct(rec_feat)

        recon_loss = self.recon_loss(x_feat, rec_feat)
        mel_loss = self.mel_loss(x, rec_x)

        return {
                "loss": recon_loss + 0.25 * vq_loss + 0.25 * mel_loss,
                "recon_loss": recon_loss,
                "vq_loss": vq_loss,
                "mel_loss": mel_loss,
                "raw_audio": x,
                "recon_audio": rec_x,
                "raw_feat": x_feat, 
                "recon_feat": rec_feat,
            }
    
    @torch.inference_mode()
    def test_one_step(self, x, x_feat, streams):
        self.eval()
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = x_feat.permute(0,3,1,2)

        enc_hs, enc_out = self.encoder.encode(x_feat)

        dec_hs, _ = self.decoder.decode(enc_hs, enc_out, streams, vqs=self.quantizer)

        rec_feat = dec_hs[-1]
        rec_x = self.audio_reconstruct(rec_feat)

        return {
                "raw_audio": x,
                "recon_audio": rec_x,
                "raw_feat": x_feat, 
                "recon_feat": rec_feat,
            }
    
    def forward(self, x, x_feat, streams, train=False):
        if train:
            return self.train_one_step(x, x_feat, streams)
        else:
            return self.test_one_step(x, x_feat, streams)
        
class BaseEncoder(nn.Module):
    def __init__(self, in_dim: int, h_dims: list) -> None:
        super().__init__()

        self.in_h_dims = [in_dim] + h_dims
        self.h_dims = h_dims

    def vis_encoder(self, enc_hs):

        for i, hs in enumerate(enc_hs):
            print(f"layer {i+1} output:", hs.shape)

class BaseCrossScaleDecoder(nn.Module):
    def __init__(self, in_H: int, in_dim: int, h_dims: list, max_streams: int, fuse_net: bool) -> None:
        super().__init__()

        self.f_dims = [in_H//2**(max_streams-1)] + \
            [in_H//2**i for i in range(max_streams-1, 0, -1)]

        self.out_h_dims = h_dims[1:] + [in_dim]
        self.h_dims = h_dims
        self.fuse_net = fuse_net
        self.max_streams = max_streams

    def vis_decoder(self, dec_hs):
        for i, hs in enumerate(dec_hs):
            print(f"layer {i+1} output:", hs.shape)
    
    def res_pre_fuse(self, enc, dec):
        return enc - dec
    def res_post_fuse(self, residual_q, dec):
        return residual_q + dec
    def conv_pre_fuse(self, enc, dec, idx):
        """enc/dec shape: [B, C, F, T]"""
        return self.pre_fuse_net[idx](torch.cat([enc,dec],dim=1))
    def conv_post_fuse(self, residual_q, dec, idx):
        """enc/dec shape: [B, C, F, T]"""
        return self.post_fuse_net[idx](torch.cat([residual_q,dec],dim=1))
    
    def w_attn_pre_fuse(self, enc, dec, idx):
        """enc/dec shape: [B, F*T, C]"""

        window_attn = self.pre_fuse_net[idx]
        aligned_dec, _ = window_attn((dec, enc))

        residual = enc - aligned_dec

        return residual
    
    def w_attn_post_fuse(self, enc, dec, idx):
        """enc/dec shape: [B, F*T, C]"""

        window_attn = self.post_fuse_net[idx]
        aligned_enc, _ = window_attn((enc, dec))

        dec_refine = dec + aligned_enc

        return  dec_refine

    def attn_pre_fuse(self, enc, dec, idx):
        """enc/dec shape: [B, F*T, C]"""
        assert enc.size() == dec.size()

        B, C = enc.size(0), enc.size(-1)
        freq = self.f_dims[idx+1]
        temp = enc.size(1) // freq

        # flatten to [B, T, C*F]
        enc1d = enc.view(B, freq, temp, C).contiguous().\
            permute(0,2,3,1).reshape(B, temp, C*freq)

        dec1d = dec.view(B, freq, temp, C).contiguous().\
            permute(0,2,3,1).reshape(B, temp, C*freq)

        mha = self.pre_fuse_net[idx]
        attn_out, attn_dist = mha(query=enc1d, key=dec1d, value=dec1d)

        residual = enc1d - attn_out

        residual = residual.view(B, temp, C, freq).contiguous().\
            permute(0,3,1,2).reshape(B, freq*temp, C)
        
        return residual

    def attn_post_fuse(self, enc, dec, idx):
        """enc/dec shape: [B, F*T, C]"""
        assert enc.size() == dec.size()
        B, C = enc.size(0), enc.size(-1)
        freq = self.f_dims[idx+1]
        temp = enc.size(1) // freq

        # flatten to [B, T, C*F]
        enc1d = enc.view(B, freq, temp, C).contiguous().\
            permute(0,2,3,1).reshape(B, temp, C*freq)

        dec1d = dec.view(B, freq, temp, C).contiguous().\
            permute(0,2,3,1).reshape(B, temp, C*freq)

        mha = self.post_fuse_net[idx]
        attn_out, attn_dist = mha(query=dec1d, key=enc1d, value=enc1d)

        dec_refine = attn_out + dec1d

        dec_refine = dec_refine.view(B, temp, C, freq).contiguous().\
            permute(0,3,1,2).reshape(B, freq*temp, C)
        return dec_refine

    def csvq_layer(self, enc, dec, idx, vq):
        if self.fuse_net is not None:
            if self.fuse_net == "conv":
                residual = self.conv_pre_fuse(enc, dec, idx)
                residual_q, vq_loss = vq(residual)
                dec_refine = self.conv_post_fuse(residual_q, dec, idx)
            
            elif self.fuse_net == "vanilla":
                residual = self.attn_pre_fuse(enc, dec, idx)
                residual_q, vq_loss = vq(residual)
                dec_refine = self.attn_post_fuse(enc, dec, idx)

            elif self.fuse_net in ["window", "shiftwindow"]:
                residual = self.w_attn_pre_fuse(enc, dec, idx)
                residual_q, vq_loss = vq(residual)
                dec_refine = self.w_attn_post_fuse(enc, dec, idx)

        else:
            residual = self.res_pre_fuse(enc, dec)
            residual_q, vq_loss = vq(residual)
            dec_refine = self.res_post_fuse(residual_q, dec)
        return dec_refine, vq_loss

class SwinEncoder(BaseEncoder):
    def __init__(self, 
                 swin_depth: int = 2,
                 swin_heads: int = 3,
                 window_size: int = 4,
                 mlp_ratio: float = 4.,
                 in_freq: int = 192, 
                 patch_size: list = [3,2], 
                 in_dim: int = 2, 
                 h_dims: list = [16,16,32,32,64,128]) -> None:
        super().__init__(in_dim, h_dims)
        
        self.patch_embed = PatchEmbed(in_freq, patch_size, in_dim, embed_dim=h_dims[0])
        self.in_h_dims = h_dims
        self.h_dims = h_dims[1:]
        self.patch_size = patch_size

        self.blocks = self.init_encoder(swin_depth, swin_heads, window_size, mlp_ratio)

        self.pre_swin = SwinTLayer(
                    self.in_h_dims[0], self.in_h_dims[0],
                    depth=swin_depth, num_heads=swin_heads[0],
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    subsample=None
        )

    def encode(self, x):
        """Step-wise Encoding with downscaling
        Args: 
            x: spectrogram feature, tensor size (B, C=2, F, T)
            returns: encoder hidden states at multiple scale
        """
        Wh, Ww = x.size(2) // self.patch_size[0], x.size(3) // self.patch_size[1]
        
        x = self.patch_embed(x)
        x, Wh, Ww = self.pre_swin(x, Wh, Ww)
        enc_hs = [x]

        for blk in self.blocks:
            x, Wh, Ww = blk(x, Wh, Ww)
            enc_hs.append(x)

        return enc_hs, Wh, Ww

    def init_encoder(self, depth, num_heads, window_size, mlp_ratio):
        blocks = nn.ModuleList()

        if len(num_heads) == 1:
            num_heads = num_heads[0]
            num_heads = [min(num_heads*2**i, num_heads*2**3) for i in range(len(self.h_dims))]
        for i in range(len(self.h_dims)):
            blocks.append(
                SwinTLayer(
                    self.in_h_dims[i],
                    self.h_dims[i],
                    depth=depth,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    subsample=PatchMerging
                )
            )
        return blocks

class SwinCrossScaleDecoder(BaseCrossScaleDecoder):
    def __init__(self, 
                 swin_depth: int = 2,
                 swin_heads: int = 3,
                 window_size: int = 4,
                 mlp_ratio: float = 4.,
                 in_freq: int = 192, 
                 patch_size: list = [3,2], 
                 in_dim: int = 2, 
                 h_dims: list = [128,64,32,32,16,16], 
                 max_streams: int = 6, 
                 fuse_net: str = "vanilla",) -> None:
        super().__init__(in_freq//patch_size[0], in_dim, h_dims, max_streams, fuse_net)

        self.patch_deembed = PatchDeEmbed(in_freq, patch_size, in_dim, h_dims[-1])
        self.h_dims = self.h_dims[:-1]
        self.out_h_dims = self.out_h_dims[:-1]
        self.blocks = self.init_decoder(swin_depth, swin_heads, window_size, mlp_ratio)

        if self.fuse_net is not None:
            print(f"Apply Residual-Based Cross Attention Fusion Net for Swin Codec | Type: {self.fuse_net}")
            if self.fuse_net == "vanilla":
                self.init_attn_fuse_blocks(max_streams, h_dims, f_dims=self.f_dims[1:], fuse_attn_heads=1)

            elif self.fuse_net == "window":
                self.init_w_attn_fuse_blocks(max_streams, h_dims=h_dims, f_dims=self.f_dims[1:], 
                                        window_size=window_size*2, fuse_attn_heads=3, shift_wa=False)
                
            elif self.fuse_net == "shiftwindow":
                self.init_w_attn_fuse_blocks(max_streams, h_dims=h_dims, f_dims=self.f_dims[1:], 
                                        window_size=window_size*2, fuse_attn_heads=3, shift_wa=True)    
                
            else:
                raise ValueError("fuse_net method must be in [vanilla, window, shiftwindow]")        
        else:
            print("Apply Vanilla Residual Fusion Net for Swin Codec")
        
        self.post_swin = SwinTLayer(
                    self.out_h_dims[-1], self.out_h_dims[-1],
                    depth=swin_depth, num_heads=swin_heads[0],
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    subsample=None
        )

    def decode(self, enc_hs: list, streams: int, vqs: nn.ModuleList, Wh: int, Ww: int):
        """Step-wise Fuse decoding (Forward Training)
        Args: 
            enc_hs: a list of encoded features at multiple scale
            streams: number of bitstreams to use <= depth + 1
            vqs: a modulelist of quantizers with size $depth$
            Wh, Ww: encoder last feature size
        """
        assert streams <= self.max_streams and len(vqs) == self.max_streams

        z0, vq_loss = vqs[0](enc_hs[-1])

        dec_hs = [z0]
        for i, blk in enumerate(self.blocks):
            transmit = (i < streams-1)            
            if transmit:
                dec_i_refine, vq_loss_i = self.csvq_layer(enc=enc_hs[-1-i], dec=dec_hs[i],
                                                          idx=i, vq=vqs[i+1])
                vq_loss += vq_loss_i
            else:
                dec_i_refine = dec_hs[i]
            
            dec_next, Wh, Ww = blk(dec_i_refine, Wh, Ww)
            dec_hs.append(dec_next)

        dec_next, Wh, Ww = self.post_swin(dec_next, Wh, Ww)
        dec_hs.append(self.patch_deembed(dec_next))
        return dec_hs, vq_loss

    def init_decoder(self, depth, num_heads, window_size, mlp_ratio):
        blocks = nn.ModuleList()
        if len(num_heads) == 1:
            num_heads = num_heads[0]
            num_heads = [min(num_heads*2**(len(self.h_dims)-i-1), num_heads*2**3) for i in range(len(self.h_dims))]
        else:
            num_heads = num_heads[::-1]
            
        for i in range(len(self.h_dims)):
            blocks.append(
                SwinTLayer(
                    self.h_dims[i],
                    self.out_h_dims[i],
                    depth=depth,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    subsample=PatchSplit
                )
            )
        return blocks
    
    def init_w_attn_fuse_blocks(self,
                                max_streams, 
                                window_size,
                                h_dims, 
                                f_dims, 
                                fuse_attn_heads=1,
                                shift_wa=False):
        
        shift_size = window_size//2
        self.pre_fuse_net, self.post_fuse_net = nn.ModuleList(), nn.ModuleList()
        for i in range(max_streams-1):
            if shift_wa:
                pre_block = nn.Sequential(*[
                    WindowAlignment(freq_size=f_dims[i], d_model=h_dims[i], 
                                    num_heads=fuse_attn_heads, window_size=window_size, shift_size=0),
                    WindowAlignment(freq_size=f_dims[i], d_model=h_dims[i],
                                    num_heads=fuse_attn_heads, window_size=window_size, shift_size=shift_size)
                    ]
                )
                post_block = nn.Sequential(*[
                    WindowAlignment(freq_size=f_dims[i], d_model=h_dims[i], 
                                    num_heads=fuse_attn_heads, window_size=window_size, shift_size=0),
                    WindowAlignment(freq_size=f_dims[i], d_model=h_dims[i],
                                    num_heads=fuse_attn_heads, window_size=window_size, shift_size=shift_size)
                ])
            else:
                pre_block = WindowAlignment(freq_size=f_dims[i], d_model=h_dims[i], 
                                            num_heads=fuse_attn_heads, window_size=window_size, shift_size=0)
                post_block = WindowAlignment(freq_size=f_dims[i], d_model=h_dims[i], 
                                            num_heads=fuse_attn_heads, window_size=window_size, shift_size=0)
            self.pre_fuse_net.append(pre_block)
            self.post_fuse_net.append(post_block)
    
    def init_attn_fuse_blocks(self, 
                         max_streams, 
                         h_dims, 
                         f_dims, 
                         fuse_attn_heads=1):
        self.pre_fuse_net, self.post_fuse_net = nn.ModuleList(), nn.ModuleList()
        for i in range(max_streams-1):
            self.pre_fuse_net.append(
                nn.MultiheadAttention(
                        embed_dim=h_dims[i]*f_dims[i],
                        num_heads=fuse_attn_heads,
                        dropout=0.0,
                        bias=False,
                        batch_first=True),
            )
            self.post_fuse_net.append(
                nn.MultiheadAttention(
                        embed_dim=h_dims[i]*f_dims[i],
                        num_heads=fuse_attn_heads,
                        dropout=0.0,
                        bias=False,
                        batch_first=True),
            )
    
class ConvEncoder(BaseEncoder):
    
    def __init__(self, in_dim: int = 2, h_dims: list = [16,16,24,24,32,64], use_tf=False) -> None:
        super().__init__(in_dim, h_dims)

        self.blocks = self.init_encoder()

        if use_tf:
            self.temp_filter_module = nn.Sequential(*[
                TCM(in_channels=6*h_dims[-1],
                    fix_channels=6*h_dims[-1]*2,
                    dilations=(1,2,4,8)),
                RNNFilter(input_size=6*h_dims[-1],
                          hidden_size=6*h_dims[-1],
                          num_layers=1,
                          batch_first=True)
            ])

            self.tf1 = TCM(in_channels=6*h_dims[-1],
                    fix_channels=6*h_dims[-1]*2,
                    dilations=(1,2,4,8))
        self.use_tf = use_tf

    def temp_filter(self, enc_last_hs):
        # B C F T
        F, T = enc_last_hs.size(2), enc_last_hs.size(3)
        input1d = enc_last_hs.reshape(enc_last_hs.size(0), -1, T) # B C*F T

        enc_out = self.temp_filter_module(input1d) # B C*F T

        enc_out = enc_out.reshape(enc_last_hs.size(0), -1, F, T) # B C F T
        return enc_out

    def encode(self, x):
        """Step-wise Encoding with downscaling
        Args: 
            x: spectrogram feature, tensor size (B, C=2, F, T)
            returns: encoder hidden states at multiple scale
        """
        enc_hs = []
        for blk in self.blocks:
            x = blk(x)
            enc_hs.append(x)

        if not self.use_tf:
            enc_out = enc_hs[-1]
        else:
            enc_out = self.temp_filter(enc_hs[-1])
            # enc_hs[-1] = enc_out

        return enc_hs, enc_out

    def init_encoder(self):
        blocks = nn.ModuleList()
        for i in range(len(self.h_dims)):
            blocks.append(
                ConvEncoderLayer(
                    self.in_h_dims[i], 
                    self.h_dims[i],
                    down = False if i == 0 else True
                )
            )
        return blocks
    

class ConvCrossScaleDecoder(BaseCrossScaleDecoder):
    def __init__(self, 
                 in_dim: int = 2, 
                 h_dims: list = [64,32,24,24,16,16],
                 fuse_net: bool = False,
                 max_streams: int = 6,
                 use_tf: bool = False
                 ) -> None:
        super().__init__(192, in_dim, h_dims, max_streams, fuse_net)

        self.blocks = self.init_decoder()

        if use_tf: 
            self.temp_filter_module = nn.Sequential(*[
                    TCM(in_channels=6*h_dims[0],
                        fix_channels=6*h_dims[0]*2,
                        dilations=(1,2,4,8)),
                    RNNFilter(input_size=6*h_dims[0],
                              hidden_size=6*h_dims[0],
                              num_layers=1,
                              batch_first=True),
                    TCM(in_channels=6*h_dims[0],
                        fix_channels=6*h_dims[0]*2,
                        dilations=(1,2,4,8))
            ])
        self.use_tf = use_tf

        if self.fuse_net:
            print("Use Attn Fuse Merge Net for Conv Codec")
            self.init_fuse_blocks(max_streams, h_dims)
        else:
            print("Use Residual Fuse for Swin Codec")

    def temp_filter(self, dec_first_hs):
        # B C F T
        F, T = dec_first_hs.size(2), dec_first_hs.size(3)
        input1d = dec_first_hs.reshape(dec_first_hs.size(0), -1, T) # B C*F T

        dec_in = self.temp_filter_module(input1d) # B C*F T
        
        dec_in = dec_in.reshape(dec_first_hs.size(0), -1, F, T) # B C F T
        return dec_in

    def decode(self, enc_hs: list, enc_out: torch.Tensor, streams: int, vqs: nn.ModuleList):
        """Step-wise Fuse decoding (Forward Training)
        Args: 
            enc_hs: a list of encoded features at multiple scale
            enc_out: last hs after temp filter (same as enc_hs[-1] if no filter applied)
            streams: number of bitstreams to use <= depth + 1
            vqs: a modulelist of quantizers with size $depth$
        """
        assert streams <= self.max_streams and len(vqs) == self.max_streams

        z0, vq_loss = vqs[0](enc_out)
        if self.use_tf:
            z0 = self.temp_filter(z0)

        dec_hs = [z0]
        for i, blk in enumerate(self.blocks):
            transmit = (i < streams-1)

            if transmit:
                dec_i_refine, vq_loss_i = self.csvq_layer(enc=enc_hs[-1-i], dec=dec_hs[i],
                                                          idx=i, vq=vqs[i+1])
                vq_loss += vq_loss_i
            else:
                dec_i_refine = dec_hs[i]

            dec_next = blk(dec_i_refine)
            dec_hs.append(dec_next)

        return dec_hs, vq_loss

    def init_decoder(self):
        blocks = nn.ModuleList()
        for i in range(len(self.h_dims)):
            blocks.append(
                ConvDecoderLayer(
                    self.h_dims[i], 
                    self.out_h_dims[i],
                    up = False if i == len(self.h_dims)-1 else True
                )
            )
        return blocks
    
    def init_fuse_blocks(self, max_streams, h_dims):
        self.pre_fuse_net, self.post_fuse_net = nn.ModuleList(), nn.ModuleList()
        for i in range(max_streams-1):
            self.pre_fuse_net.append(
                nn.Sequential(*[
                    Convolution2D(h_dims[i]*2, h_dims[i], 3, 1, 1, causal=False),
                    Convolution2D(h_dims[i], h_dims[i], 3, 1, 1, causal=False),
                ])
            )
            self.post_fuse_net.append(
                nn.Sequential(*[
                    Convolution2D(h_dims[i]*2, h_dims[i], 3, 1, 1, causal=False),
                    Convolution2D(h_dims[i], h_dims[i], 3, 1, 1, causal=False),
                ])
            )

if __name__ == "__main__":
    # codec = ConvCrossScaleCodec(fuse_net=True, use_tf=True)

    # outputs = codec.train_one_step(x=torch.randn(1,47920),
    #                      x_feat=torch.randn(1,192,600,2),
    #                      streams=6)
    
    # print(outputs["recon_feat"].shape, outputs["recon_audio"].shape)

    # outputs = codec.test_one_step(x=torch.randn(1,47920),
    #                      x_feat=torch.randn(1,192,600,2),
    #                      streams=1)
    
    codec = SwinCrossScaleCodec(patch_size = [3,2],
                                swin_depth = 2,
                                swin_heads = [3,3,6,12,24],
                                window_size = 4,
                                mlp_ratio = 4.,
                                in_dim = 2, 
                                in_freq = 192, 
                                h_dims = [45, 45, 72, 96, 192, 384], 
                                max_streams = 6, 
                                proj = [4,4,2,2,2,2], 
                                overlap = 2, 
                                num_vqs = 6, 
                                codebook_size = 1024, 
                                fuse_net="shiftwindow",
                                )
    outputs = codec.train_one_step(x=torch.randn(1,47920),
                         x_feat=torch.randn(1,192,600,2),
                         streams=6)
    # x = enc_hs[-1]
    # print(x.shape, h, w)
    # for blk in model.blocks:
    #     x, h, w = blk(x, h, w)
    #     print(x.shape, h, w)
