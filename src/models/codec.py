import torch, torchaudio, math
import torch.nn as nn
import numpy as np
from einops import rearrange

import sys
sys.path.append("/Users/tracy/Library/CloudStorage/GoogleDrive-cloudstorage.yuzhe@gmail.com/My Drive/Research/Audio_Signal_Coding/Neural-Speech-Coding/src")

from models.swin.wattn import SwinTransformerLayer, WindowAlignment
from models.swin.compress import PatchEmbed, PatchDeEmbed
from models.vq.quantization import GroupVQ, ResidualVQ, FiniteSQ
from models.losses.generator import L2Loss, MELLoss
from models.progressive_decoder import ProgressiveCrossScaleDecoder


class BaseCodec(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 in_freq: int,
                 h_dims: list,
                 max_streams: int,
                 overlap: int,
                 num_vqs: int,
                 proj_ratio: float,
                 codebook_size: int,
                 codebook_dims: list,
                 patch_size: tuple,
                 use_ema: bool,
                 use_cosine_sim: bool,
                 kmeans_init: bool,
                 mel_windows: int, 
                 mel_bins: int,
                 win_len: int, 
                 hop_len: int, 
                 sr: int,
                 vq: str,
                 ) -> None:
        super().__init__()

        self.in_freq, self.in_dim = in_freq, in_dim
        self.max_streams = max_streams

        self.stats = None
        self.enc_h_dims = h_dims
        self.dec_h_dims = h_dims[::-1]

        self.ft = torchaudio.transforms.Spectrogram(n_fft=(in_freq-1)*2, 
                                            win_length=int(win_len*sr*1e-3),
                                            hop_length=int(hop_len*sr*1e-3), power=None)
        self.ift = torchaudio.transforms.InverseSpectrogram(n_fft=(in_freq-1)*2, 
                                            win_length=int(win_len*sr*1e-3),
                                            hop_length=int(hop_len*sr*1e-3))
        
        self.mel_loss, self.recon_loss = MELLoss(mel_windows, mel_bins), L2Loss(power_law=True)
        
    def init_quantizer(self, overlap, num_vqs, proj_ratio, codebook_size, codebook_dims, 
                       patch_size=None, use_ema=True, use_cosine_sim=False, vq="GVQ", kmeans_init=False):
        quantizer = nn.ModuleList()
        if isinstance(codebook_dims, int):
            codebook_dims = [codebook_dims]*self.max_streams
        assert isinstance(codebook_dims, list), "codebook_dims should be a list"
        if len(codebook_dims)==1:
            codebook_dims = codebook_dims*self.max_streams

        if patch_size:  freq_patch, time_patch = patch_size[0], patch_size[1]
        else:   freq_patch, time_patch = 1, 1

        self.use_ema, self.use_cosine_sim = use_ema, use_cosine_sim

        quantization_map = {"GVQ": GroupVQ, "RVQ": ResidualVQ, "FSQ": FiniteSQ}
        quantizer.append(
                quantization_map[vq](
                    self.dec_h_dims[0], (self.in_freq // freq_patch) // 2**(len(self.enc_h_dims)-1), # tag
                    overlap, num_vqs, proj_ratio, codebook_dims[0], codebook_size, use_ema, use_cosine_sim, kmeans_init
                )
            ) 
        for i in range(1, self.max_streams):
            quantizer.append(
                quantization_map[vq](
                    self.dec_h_dims[i-1], (self.in_freq // freq_patch) // 2**(len(self.enc_h_dims)-i), # tag
                    overlap, num_vqs, proj_ratio, codebook_dims[i], codebook_size, use_ema, use_cosine_sim, kmeans_init
                )
            ) 
        self.max_bps = (2/overlap) * self.max_streams * math.log2(quantizer[0].codebook_size) * num_vqs // (20 * time_patch//2)
        print(f"Audio Codec {self.max_bps}kbps Initialized")
        return quantizer

    def init_quantizer_icsvq(self, overlap, num_vqs, proj_ratio, codebook_size, codebook_dims, 
                       patch_size=None, use_ema=True, use_cosine_sim=False, vq="GVQ"):
        quantizer = nn.ModuleList()
        if isinstance(codebook_dims, int):
            codebook_dims = [codebook_dims]*self.max_streams
        assert isinstance(codebook_dims, list), "codebook_dims should be a list"
        if len(codebook_dims)==1:
            codebook_dims = codebook_dims*self.max_streams

        if patch_size:  freq_patch, time_patch = patch_size[0], patch_size[1]
        else:   freq_patch, time_patch = 1, 1

        self.use_ema, self.use_cosine_sim = use_ema, use_cosine_sim

        quantization_map = {"GVQ": GroupVQ, "RVQ": ResidualVQ, "FSQ": FiniteSQ}
        for i in range(self.max_streams):
            quantizer.append(
                quantization_map[vq](
                    self.enc_h_dims[i+1], (self.in_freq // freq_patch) // 2**(i+1), # tag
                    overlap, num_vqs, proj_ratio, codebook_dims[i], codebook_size, use_ema, use_cosine_sim,
                )
            ) 
        self.max_bps = (2/overlap) * self.max_streams * math.log2(quantizer[0].codebook_size) * num_vqs // (20 * time_patch//2)
        print(f"Audio Codec {self.max_bps}kbps Initialized")
        return quantizer

    def vis_quantization(self):
        if isinstance(self.quantizer[0], ResidualVQ):   vq_type = "Residual-VQ"
        elif isinstance(self.quantizer[0], GroupVQ):    vq_type = "Group-VQ"
        else:   vq_type = "Finite-SQ"
        print("VQ-Type: ", vq_type)
        Hs, in_dims, merge_dims, each_dims, mapped_dims, codebook_dims = [], [], [], [], [], []
        for vq in self.quantizer:
            Hs.append(vq.H)
            in_dims.append(vq.in_dim)
            merge_dims.append(vq.H*vq.in_dim)
            if vq_type == "Group-VQ":
                mapped_dims.append(vq.fix_dim)
                each_dims.append(vq.fix_dim*vq.overlap // vq.num_vqs)
                codebook_dims.append(vq.vqs[0].embedding_size)
            elif vq_type == "Residual-VQ":
                mapped_dims.append(vq.H*vq.in_dim)
                each_dims.append(vq.H*vq.in_dim*vq.overlap)
                codebook_dims.append(vq.vqs[0].embedding_size)
            else:
                mapped_dims.append(vq.fix_dim)
                each_dims.append(vq.vqs.effective_codebook_dim)
                codebook_dims.append(vq.vqs.effective_codebook_dim // vq.num_vqs)
        print("Quantization Vis: ")
        print(f"     EMA: {self.use_ema} CosineSimilarity: {self.use_cosine_sim}")
        print("     Freq dims: ", Hs)
        print("     Channel(hidden) dims: ", in_dims)
        print("     Merged dims: ", merge_dims)
        print(f"     {vq_type} proj dims: ", mapped_dims)
        print(f"     {vq_type} dims (for each): ", each_dims)
        print("     Mapped Codebook dims (for each): ", codebook_dims)
    
    def spec_transform(self, x):
        """
        Args: 
            x:       audio of shape [B, L]
            returns: spectrogram of shape [B, C, H, W]
        """
         
        feat = torch.view_as_real(self.ft(x)) # B, H, W, C
        
        # Normalization
        if self.stats is not None:
            m, std = self.stats["mean"].to(x.device), self.stats["std"].to(x.device)
            feat = feat.sub(m).div(std)

        return rearrange(feat, "b h w c -> b c h w")
        
    def audio_reconstruct(self, feat):
        """
        Args: 
            feat:    spectrogram of shape [B, C, H, W]
            returns: audio of shape [B, L]
        """

        feat = torch.view_as_complex(rearrange(feat, "b c h w -> b h w c")) # B, H, W
        return self.ift(feat)
    

class SwinAudioCodec(BaseCodec):
    def __init__(self, 
                 in_dim: int = 2, 
                 in_freq: int = 192, 
                 h_dims: list = [45,45,72,96,192,384], 
                 swin_depth: int = 2,
                 swin_heads: list = [3,3,6,12,24],
                 window_size: int = 4,
                 mlp_ratio: float = 4.,
                 max_streams: int = 6, 
                 overlap: int = 2, 
                 num_vqs: int = 6, 
                 proj_ratio: float = 1.0,
                 codebook_size: int = 1024, 
                 codebook_dims: list = [8], 
                 patch_size: list = [3,2], 
                 use_ema: bool = False, 
                 use_cosine_sim: bool = True, 
                 vq: str = "GVQ",
                 kmeans_init: bool = False,
                 scalable: bool = True,
                 mel_windows: int = [32,64,128,256,512,1024,2048], 
                 mel_bins: int = [5,10,20,40,80,160,320], 
                 win_len: int = 20, 
                 hop_len: int = 5, 
                 sr: int = 16000,
                 vis: bool = True,) -> None:
        super().__init__(in_dim, in_freq, h_dims, max_streams, overlap, num_vqs, proj_ratio, codebook_size, codebook_dims, 
                    patch_size, use_ema, use_cosine_sim, kmeans_init, mel_windows, mel_bins, win_len, hop_len, sr, vq)

        self.scalable = scalable
        self.quantizer = self.init_quantizer(
                        overlap, num_vqs, proj_ratio, codebook_size, codebook_dims,
                        patch_size, use_ema, use_cosine_sim, vq, kmeans_init)

        self.encoder = SwinEncoder(
            swin_depth, swin_heads, window_size, mlp_ratio,
            in_freq, patch_size, self.in_dim, self.enc_h_dims, False
        )
        self.decoder = SwinCrossScaleDecoder(
            swin_depth, swin_heads, window_size, mlp_ratio,
            in_freq, patch_size, self.in_dim, self.dec_h_dims, False,
            max_streams, "None",
        )
        if vis:
            print("Apply Vanilla Residual Fusion Net for Swin Codec")
            self.vis_quantization()
            self.vis_encoder_decoder()

    def train_one_step(self, x, x_feat=None, streams=6, freeze_codebook=False):
        self.train()
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = x_feat.permute(0,3,1,2)

        enc_hs, Wh, Ww = self.encoder.encode(x_feat)
        recon_feat, _, cm_loss, cb_loss, kl_loss, multi_codes = self.decoder.decode(
                enc_hs, streams, vqs=self.quantizer, Wh=Wh, Ww=Ww, freeze_codebook=freeze_codebook)
        rec_x = self.audio_reconstruct(recon_feat)

        recon_loss = self.recon_loss(x_feat, recon_feat)
        mel_loss = self.mel_loss(x, rec_x)

        return {
                "recon_loss": recon_loss,
                "commitment_loss": cm_loss,
                "codebook_loss": cb_loss,
                "mel_loss": mel_loss,
                "kl_loss": kl_loss,
                "raw_audio": x,
                "recon_audio": rec_x,
                "raw_feat": x_feat, 
                "recon_feat": recon_feat,
            }
    
    @torch.inference_mode()
    def test_one_step(self, x, x_feat, streams, freeze_codebook=False):
        self.eval()
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = x_feat.permute(0,3,1,2)

        enc_hs, Wh, Ww = self.encoder.encode(x_feat)

        recon_feat, _, cm_loss, cb_loss, kl_loss, multi_codes = self.decoder.decode(
            enc_hs, streams, vqs=self.quantizer, Wh=Wh, Ww=Ww, freeze_codebook=freeze_codebook)

        rec_x = self.audio_reconstruct(recon_feat)
        return {
                "raw_audio": x,
                "recon_audio": rec_x,
                "raw_feat": x_feat, 
                "recon_feat": recon_feat,
                "multi_codes": multi_codes,
            }
    
    def forward(self, x, x_feat, streams, train=False, freeze_codebook=False):
        """
        Args: 
            x: audio Tensor
            x_feat: audio complex STFT
            streams: number of streams transmitted
            train: set False in inference
            freeze_codebook: boolean set True only during pretrain stage (meanwhile streams has to be max_stream)
        """
        streams = self.max_streams if freeze_codebook else streams
        if train:
            return self.train_one_step(x, x_feat, streams, freeze_codebook)
        else:
            return self.test_one_step(x, x_feat, streams, freeze_codebook)
        
    @torch.inference_mode()
    def encode(self, x, num_streams=6):
        self.eval()

        x_feat = self.spec_transform(x)
        enc_hs, Wh, Ww = self.encoder.encode(x_feat)

        multi_codes = self.decoder.quantize(enc_hs, num_streams, vqs=self.quantizer, Wh=Wh, Ww=Ww)
        return multi_codes, (Wh, Ww)
    
    @torch.inference_mode()
    def decode(self, multi_codes, enc_feat_size=(2, 300)):
        self.eval()

        Wh, Ww = enc_feat_size
        dec_hs = self.decoder.dequantize(multi_codes, vqs=self.quantizer, Wh=Wh, Ww=Ww)

        rec_feat = dec_hs[-1]
        rec_x = self.audio_reconstruct(rec_feat)
        return rec_x
    
    def vis_encoder_decoder(self):
        self.encoder.vis_encoder()
        print()
        self.decoder.vis_decoder()


class CrossScaleProgressiveResCodec(BaseCodec):
    def __init__(self, in_dim: int = 2, 
                 in_freq: int = 192, 
                 h_dims: list = [45,45,72,96,192,384], 
                 swin_depth: int = 2,
                 swin_heads: list = [3,3,6,12,24],
                 window_size: int = 4,
                 mlp_ratio: float = 4.,
                 max_streams: int = 6, 
                 overlap: int = 2, 
                 num_vqs: int = 6, 
                 proj_ratio: float = 1.0,
                 codebook_size: int = 1024, 
                 codebook_dims: list = [8], 
                 patch_size: list = [3,2], 
                 use_ema: bool = False, 
                 use_cosine_sim: bool = True, 
                 vq: str = "GVQ",
                 scalable: bool = True,
                 mel_windows: int = [32,64,128,256,512,1024,2048], 
                 mel_bins: int = [5,10,20,40,80,160,320], 
                 win_len: int = 20, 
                 hop_len: int = 5, 
                 sr: int = 16000,
                 vis: bool = True,) -> None:
        super().__init__(in_dim, in_freq, h_dims, max_streams, overlap, num_vqs, proj_ratio, codebook_size, codebook_dims, 
                patch_size, use_ema, use_cosine_sim, mel_windows, mel_bins, win_len, hop_len, sr, vq)

        self.scalable = scalable
        self.encoder = SwinEncoder(
            swin_depth, swin_heads, window_size, mlp_ratio,
            in_freq, patch_size, self.in_dim, self.enc_h_dims, False
        )
        self.decoder = ProgressiveCrossScaleDecoder(
            swin_depth, swin_heads, window_size, mlp_ratio, in_freq, patch_size, 
            self.in_dim, self.dec_h_dims, is_causal=False, max_streams=max_streams
        )

        if vis:
            self.vis_quantization()
            self.vis_encoder_decoder()

    def train_one_step(self, x, x_feat=None, streams=6, alpha=1.0):
        """ Progressive Training one step forward
        Args: 
            x: audio Tensor with shape [bs, L]
            x_feat: optional transformed complex audio STFT with shape [bs, 2, F, T]
            streams: number of cross scale codebooks to use
            alpha: hyperparameter for blending new layers in progressive training ranged in [0,1]
        """
        self.train()
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = x_feat.permute(0,3,1,2)

        enc_hs, Wh, Ww = self.encoder.encode(x_feat)
        recon_feat, _, cm_loss, cb_loss, kl_loss = self.decoder.decode(enc_hs, streams, vqs=self.quantizer, Wh=Wh, Ww=Ww, alpha=alpha)
        rec_x = self.audio_reconstruct(recon_feat)

        recon_loss = self.recon_loss(x_feat, recon_feat)
        mel_loss = self.mel_loss(x, rec_x)

        return {    "recon_loss": recon_loss,
                    "commitment_loss": cm_loss,
                    "codebook_loss": cb_loss,
                    "mel_loss": mel_loss,
                    "kl_loss": kl_loss,
                    "raw_audio": x,
                    "recon_audio": rec_x,
                    "raw_feat": x_feat, 
                    "recon_feat": recon_feat, }
    
    @torch.inference_mode()
    def test_one_step(self, x, x_feat, streams):
        """ Inference one step forward
        Args: 
            x: audio Tensor with shape [bs, L]
            x_feat: optional transformed complex audio STFT with shape [bs, 2, F, T]
            streams: number of cross scale codebooks to use
        """
        self.eval()
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = x_feat.permute(0,3,1,2)

        enc_hs, Wh, Ww = self.encoder.encode(x_feat)

        recon_feat, _, cm_loss, cb_loss, kl_loss = self.decoder.decode(enc_hs, streams, vqs=self.quantizer, Wh=Wh, Ww=Ww, alpha=1.0)
        rec_x = self.audio_reconstruct(recon_feat)

        return {    "raw_audio": x,
                    "recon_audio": rec_x,
                    "raw_feat": x_feat, 
                    "recon_feat": recon_feat    }
    
    def forward(self, x, x_feat, streams, alpha, train=False):
        if train:
            return self.train_one_step(x, x_feat, streams, alpha)
        else:
            return self.test_one_step(x, x_feat, streams)
        
    @torch.inference_mode()
    def encode(self, x, num_streams=6):
        self.eval()

        x_feat = self.spec_transform(x)
        enc_hs, Wh, Ww = self.encoder.encode(x_feat)

        multi_codes = self.decoder.quantize(enc_hs, num_streams, vqs=self.quantizer, Wh=Wh, Ww=Ww)
        return multi_codes, (Wh, Ww)
    
    @torch.inference_mode()
    def decode(self, multi_codes, enc_feat_size=(2, 300)):
        self.eval()

        Wh, Ww = enc_feat_size
        dec_hs = self.decoder.dequantize(multi_codes, vqs=self.quantizer, Wh=Wh, Ww=Ww)

        rec_feat = dec_hs[-1]
        rec_x = self.audio_reconstruct(rec_feat)
        return rec_x

    def vis_encoder_decoder(self):
        self.encoder.vis_encoder()
        print()
        self.decoder.vis_decoder()
       

class BaseEncoder(nn.Module):
    def __init__(self, in_dim: int, h_dims: list) -> None:
        super().__init__()

        self.in_h_dims = [in_dim] + h_dims
        self.h_dims = h_dims


class SwinEncoder(BaseEncoder):
    def __init__(self, 
                 swin_depth: int = 2,
                 swin_heads: list = [3],
                 window_size: int = 4,
                 mlp_ratio: float = 4.,
                 in_freq: int = 192, 
                 patch_size: list = [3,2], 
                 in_dim: int = 2, 
                 h_dims: list = [45,45,72,96,192,384],
                 is_causal: bool = False) -> None:
        super().__init__(in_dim, h_dims)
        
        self.patch_embed = PatchEmbed(in_freq, patch_size, in_dim, embed_dim=h_dims[0])
        self.in_h_dims = h_dims
        self.h_dims = h_dims[1:]
        self.patch_size = patch_size

        self.blocks = self.init_encoder(swin_depth, swin_heads, window_size, mlp_ratio, is_causal)
        self.pre_swin = SwinTransformerLayer(
                    self.in_h_dims[0], self.in_h_dims[0],
                    depth=swin_depth, num_heads=swin_heads[0],
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    subsample=None, is_causal=is_causal,
        )# tag

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

    def init_encoder(self, depth, num_heads: list, window_size, mlp_ratio, is_causal):

        if len(num_heads) == 1:
            num_heads = num_heads[0]
            num_heads = [min(num_heads*2**i, num_heads*2**3) for i in range(len(self.h_dims))]

        blocks = nn.ModuleList()
        for i in range(len(self.h_dims)):
            blocks.append(
                SwinTransformerLayer(
                    self.in_h_dims[i],
                    self.h_dims[i],
                    depth=depth,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    subsample="down",
                    scale_factor=(2,1),
                    is_causal=is_causal,
                )
            )
        return blocks
    
    def vis_encoder(self):
        
        blk = self.pre_swin
        print("Pre-swin Layer: swin_depth={} swin_hidden={} heads={} down={}".format(
                blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))
        
        for i, blk in enumerate(self.blocks):

            print("Layer[{}]: swin_depth={} swin_hidden={} heads={} down={}".format(
                i, blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))
    


class BaseCrossScaleDecoder(nn.Module):
    def __init__(self, in_H: int, in_dim: int, h_dims: list, max_streams: int, fuse_net: str,) -> None:
        super().__init__()

        # self.f_dims = [in_H//2**(max_streams-1)] + \
        #     [in_H//2**i for i in range(max_streams-1, 0, -1)]

        self.out_h_dims = h_dims[1:] + [in_dim]
        self.h_dims = h_dims
        self.fuse_net = fuse_net
        self.max_streams = max_streams

        if self.fuse_net in ["None"]:
            self.pre_fuse, self.post_fuse = self.res_pre_fuse, self.res_post_fuse
    
    def res_pre_fuse(self, enc, dec, idx=None, pre_fuse_net=None, Wh=None, Ww=None):
        return pre_fuse_net[idx](enc - dec)
    
    def res_post_fuse(self, residual_q, dec, idx=None, post_fuse_net=None, Wh=None, Ww=None, transmit=True):
        if not transmit: 
            mask = (
                torch.full((dec.shape[0],), fill_value=False, device=dec.device)
            )
            residual_q *= mask[:, None, None]
        return post_fuse_net[idx](residual_q + dec)

    def csvq_layer(self, 
                   enc: torch.tensor, 
                   dec: torch.tensor, 
                   idx: int, 
                   vq: nn.Module, 
                   pre_fuse_net: nn.ModuleList = None, 
                   post_fuse_net: nn.ModuleList = None, 
                   Wh: int = None, 
                   Ww: int = None,
                   transmit: bool=True,
                   freeze_codebook: bool=False):
        # Quantization Forward that combines quantize and dequantize
        residual = self.pre_fuse(enc, dec, idx, pre_fuse_net, Wh, Ww)
        residual_q, cm_loss, cb_loss, kl_loss, indices, _ = vq(residual, freeze_codebook)
        dec_refine = self.post_fuse(residual_q, dec, idx, post_fuse_net, Wh, Ww, transmit)

        if not transmit:
            mask = torch.full((dec.shape[0],), fill_value=False, device=dec.device)
            cm_loss *= mask
            cb_loss *= mask
            kl_loss *= mask

        return dec_refine, cm_loss, cb_loss, kl_loss, indices
    
    def csvq_quantize(self, enc, dec, idx, vq, pre_fuse_net, Wh:int, Ww:int):

        residual = self.pre_fuse(enc, dec, idx, pre_fuse_net, Wh, Ww)
        codes = vq.encode(residual)
        return codes
    
    def csvq_dequantize(self, codes, dec, idx, vq, post_fuse_net, Wh:int, Ww:int):

        residual_q = vq.decode(codes, dim=3) # dim=3 for transformer / dim=4 for convolution
        dec_refine = self.post_fuse(residual_q, dec, idx, post_fuse_net, Wh, Ww)
        return dec_refine
    

class SwinCrossScaleDecoder(BaseCrossScaleDecoder):
    def __init__(self, 
                 swin_depth: int = 2,
                 swin_heads: list = [3],
                 window_size: int = 4,
                 mlp_ratio: float = 4.,
                 in_freq: int = 192, 
                 patch_size: list = [3,2], 
                 in_dim: int = 2, 
                 h_dims: list = [384,192,96,72,45,45], 
                 is_causal: bool = False,
                 max_streams: int = 6, 
                 fuse_net: str = "vanilla",) -> None:
        super().__init__(in_freq//patch_size[0], in_dim, h_dims, max_streams, fuse_net,)

        self.patch_deembed = PatchDeEmbed(in_freq, patch_size, in_dim, h_dims[-1])
        self.h_dims = self.h_dims[:-1]
        self.out_h_dims = self.out_h_dims[:-1]
        self.blocks = self.init_decoder(swin_depth, swin_heads, window_size, mlp_ratio, is_causal)

        pre_fuse_net = nn.ModuleList([nn.Identity() for _ in range(max_streams-1)])
        post_fuse_net = nn.ModuleList([nn.Identity() for _ in range(max_streams-1)])
        self.pre_fuse_net, self.post_fuse_net = pre_fuse_net, post_fuse_net

        self.post_swin = SwinTransformerLayer(
                    self.out_h_dims[-1], self.out_h_dims[-1],
                    depth=swin_depth, num_heads=swin_heads[0],
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    subsample=None, is_causal=is_causal
        )# tag

    def decode(self, enc_hs: list, streams: int, vqs: nn.ModuleList, Wh: int, Ww: int, freeze_codebook: bool=False):
        """Step-wise Fuse decoding (Combines Quantize and Dequantize for Forward Training)
        Args: 
            enc_hs: a list of encoded features at multiple scale
            streams: number of bitstreams to use <= depth + 1
            vqs: a modulelist of quantizers with size $depth$
            Wh, Ww: encoder last feature size
            freeze_codebook: boolean (True when no codebook is used in pretraining, in which streams=6)
        """
        assert streams <= self.max_streams and len(vqs) == self.max_streams

        z0, cm_loss, cb_loss, kl_loss, indices, _ = vqs[0](enc_hs[-1], freeze_codebook)
        multi_codes = [indices]     # [(bs,G,T)]*streams

        dec_hs = [z0]
        for i, blk in enumerate(self.blocks):
            transmit = (i < streams-1)    
            if self.training:
                dec_i_refine, cm_loss_i, cb_loss_i, kl_loss_i, indices_i = self.csvq_layer(
                                                    enc=enc_hs[-1-i], dec=dec_hs[i],
                                                    idx=i, vq=vqs[i+1], 
                                                    pre_fuse_net=self.pre_fuse_net,
                                                    post_fuse_net=self.post_fuse_net,
                                                    Wh=Wh, Ww=Ww, transmit=transmit, freeze_codebook=freeze_codebook)
                cm_loss += cm_loss_i
                cb_loss += cb_loss_i
                kl_loss += kl_loss_i
                if indices_i is not None:
                    multi_codes.append(indices_i)
            else:
                if transmit:
                    dec_i_refine, cm_loss_i, cb_loss_i, kl_loss_i, indices_i = self.csvq_layer(
                                                            enc=enc_hs[-1-i], dec=dec_hs[i],
                                                            idx=i, vq=vqs[i+1], 
                                                            pre_fuse_net=self.pre_fuse_net,
                                                            post_fuse_net=self.post_fuse_net,
                                                            Wh=Wh, Ww=Ww, freeze_codebook=freeze_codebook)
                    cm_loss += cm_loss_i
                    cb_loss += cb_loss_i
                    kl_loss += kl_loss_i
                    if indices_i is not None:
                        multi_codes.append(indices_i)
                else:
                    dec_i_refine = dec_hs[i]
            
            dec_next, Wh, Ww = blk(dec_i_refine, Wh, Ww)
            dec_hs.append(dec_next)

        dec_next, Wh, Ww = self.post_swin(dec_next, Wh, Ww)
        recon_feat = self.patch_deembed(dec_next)

        return recon_feat, dec_hs, cm_loss, cb_loss, kl_loss, multi_codes
    
    def quantize(self, enc_hs: list, streams: int, vqs: nn.ModuleList, Wh: int, Ww: int):
        """Step-wise Compression (Quantize to code for Inference)
        Args: 
            enc_hs: a list of encoded features at multiple scale
            streams: number of bitstreams to use <= depth + 1
            vqs: a modulelist of quantizers with size $depth$
            Wh, Ww: encoder last feature size
        returns: multi-scale codes
        """
        assert streams <= self.max_streams and len(vqs) == self.max_streams

        codes0 = vqs[0].encode(enc_hs[-1])
        if streams == 1:
            return [codes0]
        
        z0 = vqs[0].decode(codes0)
        multi_codes, dec_hs = [codes0], [z0]
        for i in range(streams-1):
            
            codes_i = self.csvq_quantize(enc=enc_hs[-1-i], dec=dec_hs[i], idx=i, vq=vqs[i+1], pre_fuse_net=self.pre_fuse_net, Wh=Wh, Ww=Ww)
            multi_codes.append(codes_i)
            if len(multi_codes) == streams: 
                break
            dec_i_refine = self.csvq_dequantize(codes=codes_i, dec=dec_hs[i], idx=i, vq=vqs[i+1], post_fuse_net=self.post_fuse_net, Wh=Wh, Ww=Ww)

            dec_next, Wh, Ww = self.blocks[i](dec_i_refine, Wh, Ww)
            dec_hs.append(dec_next)

        return multi_codes
    
    def dequantize(self, multi_codes: list, vqs: nn.ModuleList, Wh: int, Ww: int):
        """Step-wise DeCompression (DeQuantize code for Inference)
        Args: 
            multi_codes: a list of encoded residual codes at multiple scale
            vqs: a modulelist of quantizers with size $depth$
            Wh, Ww: encoder last feature size
        returns: multi-scale codes
        """
        streams = len(multi_codes)
        assert streams <= self.max_streams and len(vqs) == self.max_streams

        z0 = vqs[0].decode(multi_codes[0])
        dec_hs = [z0]
        for i in range(streams-1): # Using code of residuals to refine decoding
            dec_i_refine = self.csvq_dequantize(codes=multi_codes[i+1], dec=dec_hs[i], idx=i, vq=vqs[i+1], post_fuse_net=self.post_fuse_net, Wh=Wh, Ww=Ww)

            dec_next, Wh, Ww = self.blocks[i](dec_i_refine, Wh, Ww)
            dec_hs.append(dec_next)

        dec_next, Wh, Ww = self.post_swin(dec_next, Wh, Ww)
        dec_hs.append(self.patch_deembed(dec_next))
        return dec_hs

    def init_decoder(self, depth, num_heads: list, window_size, mlp_ratio, is_causal):
       
        if len(num_heads) == 1:
            num_heads = num_heads[0]
            num_heads = [min(num_heads*2**(len(self.h_dims)-i-1), num_heads*2**3) for i in range(len(self.h_dims))]
        else:
            num_heads = num_heads[::-1]

        blocks = nn.ModuleList()
        for i in range(len(self.h_dims)):
            blocks.append(
                SwinTransformerLayer(
                    self.h_dims[i],
                    self.out_h_dims[i],
                    depth=depth,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    subsample="up",
                    scale_factor=[2,1],
                    is_causal=is_causal
                )
            )
        return blocks

    def vis_decoder(self):
        
        for i, blk in enumerate(self.blocks):

            print("Layer[{}]: swin_depth={} swin_hidden={} heads={} up={}".format(
                i, blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))

        blk = self.post_swin
        print("Post-swin Layer: swin_depth={} swin_hidden={} heads={} up={}".format(
                blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))


class SwinDecoder(nn.Module):
    def __init__(self, 
                 swin_depth: int = 2,
                 swin_heads: list = [3],
                 window_size: int = 4,
                 mlp_ratio: float = 4.,
                 in_freq: int = 192, 
                 patch_size: list = [3,2], 
                 in_dim: int = 2, 
                 h_dims: list = [384,192,96,72,45,45], 
                 is_causal: bool = False,) -> None:
        super().__init__()

        self.patch_deembed = PatchDeEmbed(in_freq, patch_size, in_dim, h_dims[-1])
        self.h_dims = h_dims[:-1]
        self.out_h_dims = h_dims[1:]
        self.blocks = self.init_decoder(swin_depth, swin_heads, window_size, mlp_ratio, is_causal)

        self.post_swin = SwinTransformerLayer(
                    self.out_h_dims[-1], self.out_h_dims[-1],
                    depth=swin_depth, num_heads=swin_heads[0],
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    subsample=None, is_causal=is_causal
        )# tag

    def init_decoder(self, depth, num_heads: list, window_size, mlp_ratio, is_causal):
       
        if len(num_heads) == 1:
            num_heads = num_heads[0]
            num_heads = [min(num_heads*2**(len(self.h_dims)-i-1), num_heads*2**3) for i in range(len(self.h_dims))]
        else:
            num_heads = num_heads[::-1]

        blocks = nn.ModuleList()
        for i in range(len(self.h_dims)):
            blocks.append(
                SwinTransformerLayer(
                    self.h_dims[i],
                    self.out_h_dims[i],
                    depth=depth,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    subsample="up",
                    scale_factor=[2,1],
                    is_causal=is_causal
                )
            )
        return blocks

if __name__ == "__main__":
    import os, yaml
    from utils import dict2namespace
    with open(os.path.join('configs', 'residual_9k_fsq.yml'), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    model = SwinAudioCodec(config.model.in_dim, config.model.in_freq, config.model.h_dims,
                    config.model.swin_depth, config.model.swin_heads, config.model.window_size, 
                    config.model.mlp_ratio, config.model.max_streams, config.model.overlap, 
                    config.model.num_vqs, config.model.proj_ratio, config.model.codebook_size, config.model.codebook_dims,
                    config.model.patch_size, config.model.use_ema, config.model.use_cosine_sim, config.model.vq, config.model.scalable,
                    config.model.mel_windows, config.model.mel_bins, config.model.win_len,
                    config.model.hop_len, config.model.sr, vis=True)
    trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
    )
    print(trainable_params)

    # x_feat_ = torch.ones(2,2,192,600)
    # enc_hs, Wh, Ww = model.encoder.encode(x_feat_)
    # for i in range(len(enc_hs)):
    #     print(enc_hs[i].shape)
    
    # model.train()
    # recon_feat, _, cm_loss, cb_loss, dec_refines = model.decoder.decode(enc_hs, 2, vqs=model.quantizer, Wh=Wh, Ww=Ww)
    # model.eval()
    # recon_feat_, _, cm_loss_, cb_loss_, dec_refines_ = model.decoder.decode(enc_hs, 2, vqs=model.quantizer, Wh=Wh, Ww=Ww)

    # print((recon_feat_-recon_feat).sum(), (cm_loss-cm_loss_).sum(), (cb_loss-cb_loss_).sum())

    # for i in range(len(dec_refines)):
    #     print(i, (dec_refines_[i]-dec_refines[i]).sum())

    outputs = model.train_one_step(x=torch.ones(2,47920),
                         x_feat=torch.randn(2,192,600,2),
                         streams=0)

    for k, v in outputs.items():
        if "loss" in k:
            print(k, v)
    
    # for vq in model.quantizer:
    #     print(vq)
    # codes, _ = model.encode(x=torch.ones(2, 47920), num_streams=6)
    # print(len(codes), codes[0].shape)

    # print(model.decode(codes, enc_feat_size=(1, 150)))
        
        
