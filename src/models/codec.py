import torch, torchaudio, math
import torch.nn as nn
import numpy as np
from einops import rearrange

import sys
sys.path.append("/Users/tracy/Library/CloudStorage/GoogleDrive-cloudstorage.yuzhe@gmail.com/My Drive/Research/Audio_Signal_Coding/Neural-Speech-Coding/src")

from models.swin.wattn import SwinTransformerLayer, WindowAlignment
from models.swin.compress import PatchEmbed, PatchDeEmbed
from models.vq.quantization import GroupVQ, ResidualVQ
from models.losses.generator import L2Loss, MELLoss


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
                 mel_windows: int, 
                 mel_bins: int,
                 win_len: int, 
                 hop_len: int, 
                 sr: int,
                 augment: bool,
                 use_rvq: bool,
                 ) -> None:
        super().__init__()

        self.in_freq, self.in_dim = in_freq, in_dim
        self.max_streams = max_streams

        self.stats = None
        self.freq_augment = torchaudio.transforms.FrequencyMasking(freq_mask_param=80, iid_masks=True) if augment else None
        self.time_augment = torchaudio.transforms.TimeMasking(time_mask_param=80, iid_masks=True) if augment else None

        self.enc_h_dims = h_dims
        self.dec_h_dims = h_dims[::-1]

        self.ft = torchaudio.transforms.Spectrogram(n_fft=(in_freq-1)*2, 
                                            win_length=int(win_len*sr*1e-3),
                                            hop_length=int(hop_len*sr*1e-3), power=None)
        self.ift = torchaudio.transforms.InverseSpectrogram(n_fft=(in_freq-1)*2, 
                                            win_length=int(win_len*sr*1e-3),
                                            hop_length=int(hop_len*sr*1e-3))
        
        self.quantizer = self.init_quantizer(overlap, num_vqs, proj_ratio, codebook_size, codebook_dims,
                            patch_size, use_ema, use_cosine_sim, use_rvq)
        self.use_ema, self.use_cosine_sim = use_ema, use_cosine_sim
        
        self.mel_loss = MELLoss(mel_windows, mel_bins)
        self.recon_loss = L2Loss(power_law=True)
        
    def init_quantizer(self, overlap, num_vqs, proj_ratio, codebook_size, codebook_dims, 
                       patch_size=None, use_ema=True, use_cosine_sim=False, use_rvq=False):
        quantizer = nn.ModuleList()

        if isinstance(codebook_dims, int):
            codebook_dims = [codebook_dims]*self.max_streams
        assert isinstance(codebook_dims, list), "codebook_dims should be a list"
        if len(codebook_dims)==1:
            codebook_dims = codebook_dims*self.max_streams

        if patch_size:
            freq_patch, time_patch = patch_size[0], patch_size[1]
        else:
            freq_patch, time_patch = 1, 1

        quantizer.append(
                GroupVQ(
                    self.dec_h_dims[0], (self.in_freq // freq_patch) // 2**(len(self.enc_h_dims)-1), # tag
                    overlap, num_vqs, proj_ratio, codebook_dims[0], codebook_size, use_ema, use_cosine_sim
                ) if not use_rvq else \
                ResidualVQ(
                    self.dec_h_dims[0], (self.in_freq // freq_patch) // 2**(len(self.enc_h_dims)-1),
                    overlap, num_vqs, codebook_dims[0], codebook_size, use_ema, use_cosine_sim
                )
            ) 
        for i in range(1, self.max_streams):
            quantizer.append(
                GroupVQ(
                    self.dec_h_dims[i-1], (self.in_freq // freq_patch) // 2**(len(self.enc_h_dims)-i), # tag
                    overlap, num_vqs, proj_ratio, codebook_dims[i], codebook_size, use_ema, use_cosine_sim,
                ) if not use_rvq else \
                ResidualVQ(
                    self.dec_h_dims[i-1], (self.in_freq // freq_patch) // 2**(len(self.enc_h_dims)-i),
                    overlap, num_vqs, codebook_dims[i], codebook_size, use_ema, use_cosine_sim
                )
            ) 
        self.max_bps = (2/overlap) * self.max_streams * math.log2(codebook_size) * num_vqs // (20 * time_patch//2)
        print(f"Audio Codec {self.max_bps}kbps Initialized")
        return quantizer
    
    def vis_quantization(self):
        vq_type = "Residual-VQ" if isinstance(self.quantizer[0], ResidualVQ) else "Group-VQ"
        print("VQ-Type: ", vq_type)
        Hs, in_dims, merge_dims, each_dims, mapped_dims, codebook_dims = [], [], [], [], [], []
        for vq in self.quantizer:
            Hs.append(vq.H)
            in_dims.append(vq.in_dim)
            merge_dims.append(vq.H*vq.in_dim)
            if vq_type == "Group-VQ":
                mapped_dims.append(vq.fix_dim)
                each_dims.append(vq.fix_dim*vq.overlap // vq.num_vqs)
            elif vq_type == "Residual-VQ":
                mapped_dims.append(vq.H*vq.in_dim)
                each_dims.append(vq.H*vq.in_dim*vq.overlap)
            codebook_dims.append(vq.vqs[0].embedding_size)
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
                 is_causal: bool = False,
                 use_rvq: bool = False,
                 fuse_net: str = None,
                 scalable: bool = True,
                 augment: bool = False,
                 mel_windows: int = [32,64,128,256,512,1024,2048], 
                 mel_bins: int = [5,10,20,40,80,160,320], 
                 win_len: int = 20, 
                 hop_len: int = 5, 
                 sr: int = 16000,
                 vis: bool = True,) -> None:
        super().__init__(in_dim, in_freq, h_dims, max_streams, overlap, num_vqs, proj_ratio, codebook_size, codebook_dims, patch_size, use_ema, use_cosine_sim, mel_windows, mel_bins, win_len, hop_len, sr, augment, use_rvq)

        self.scalable = scalable
        self.fuse_net = fuse_net

        self.encoder = SwinEncoder(
            swin_depth, swin_heads, window_size, mlp_ratio,
            in_freq, patch_size, self.in_dim, self.enc_h_dims, is_causal
        )
        self.decoder = SwinCrossScaleDecoder(
            swin_depth, swin_heads, window_size, mlp_ratio,
            in_freq, patch_size, self.in_dim, self.dec_h_dims, is_causal,
            max_streams, fuse_net,
        )

        if vis:
            print("Apply Vanilla Residual Fusion Net for Swin Codec")
            self.vis_quantization()
            self.vis_encoder_decoder()

    def train_one_step(self, x, x_feat=None, streams=6):
        self.train()
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = x_feat.permute(0,3,1,2)

        # x_feat_ = x_feat.clone()
        # if self.time_augment is not None:
        #     x_feat_[:, 0, :, :] = self.time_augment(x_feat_[:, 0, :, :])
        #     x_feat_[:, 1, :, :] = self.time_augment(x_feat_[:, 1, :, :])
        # if self.freq_augment is not None:
        #     x_feat_[:, 0, :, :] = self.freq_augment(x_feat_[:, 0, :, :])
        #     x_feat_[:, 1, :, :] = self.freq_augment(x_feat_[:, 1, :, :])

        enc_hs, Wh, Ww = self.encoder.encode(x_feat)
        recon_feat, _, cm_loss, cb_loss, kl_loss = self.decoder.decode(enc_hs, streams, vqs=self.quantizer, Wh=Wh, Ww=Ww)
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
    def test_one_step(self, x, x_feat, streams):
        self.eval()
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = x_feat.permute(0,3,1,2)

        enc_hs, Wh, Ww = self.encoder.encode(x_feat)

        recon_feat, _, cm_loss, cb_loss, kl_loss = self.decoder.decode(enc_hs, streams, vqs=self.quantizer, Wh=Wh, Ww=Ww)

        rec_x = self.audio_reconstruct(recon_feat)

        return {
                "raw_audio": x,
                "recon_audio": rec_x,
                "raw_feat": x_feat, 
                "recon_feat": recon_feat,
            }
    
    def forward(self, x, x_feat, streams, train=False):
        if train:
            return self.train_one_step(x, x_feat, streams)
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

        self.f_dims = [in_H//2**(max_streams-1)] + \
            [in_H//2**i for i in range(max_streams-1, 0, -1)]

        self.out_h_dims = h_dims[1:] + [in_dim]
        self.h_dims = h_dims
        self.fuse_net = fuse_net
        self.max_streams = max_streams

        if self.fuse_net in ["None"]:
            self.pre_fuse, self.post_fuse = self.res_pre_fuse, self.res_post_fuse

        elif self.fuse_net in ["cross_w_merge", "cross_sw_merge"]:
            self.pre_fuse, self.post_fuse = self.cross_w_attn_pre_fuse, self.cross_w_attn_post_fuse

        elif self.fuse_net in ["self_w_merge", "self_sw_merge"]:
            self.pre_fuse, self.post_fuse = self.self_w_attn_pre_fuse, self.self_w_attn_post_fuse
    
    def res_pre_fuse(self, enc, dec, idx=None, pre_fuse_net=None, Wh=None, Ww=None):
        return pre_fuse_net[idx](enc - dec)
    
    def res_post_fuse(self, residual_q, dec, idx=None, post_fuse_net=None, Wh=None, Ww=None, transmit=True):
        if not transmit: 
            mask = (
                torch.full((dec.shape[0],), fill_value=False, device=dec.device)
            )
            residual_q *= mask[:, None, None]
        return post_fuse_net[idx](residual_q + dec)
    
    def cross_w_attn_pre_fuse(self, enc, dec, idx, pre_fuse_net, Wh=None, Ww=None):
        """enc/dec shape: [B, F*T, C]"""

        window_attn = pre_fuse_net[idx]
        aligned_dec, _ = window_attn((dec, enc))
        residual = enc - aligned_dec

        return residual
    def cross_w_attn_post_fuse(self, residual_q, dec, idx, post_fuse_net, Wh=None, Ww=None):
        """residual_q/dec shape: [B, F*T, C]"""

        window_attn = post_fuse_net[idx]
        aligned_enc, _ = window_attn((residual_q, dec))
        dec_refine = dec + aligned_enc

        return dec_refine
    
    def self_w_attn_pre_fuse(self, enc, dec, idx, pre_fuse_net, Wh, Ww):
        """enc/dec shape: [B, F*T, C]"""

        window_attn = pre_fuse_net[idx]
        merge = torch.cat((enc,dec), dim=-1)

        merge, Wh, Ww = window_attn(merge, Wh, Ww)

        return merge
    
    def self_w_attn_post_fuse(self, residual_q, dec, idx, post_fuse_net, Wh, Ww):
        """residual_q/dec shape: [B, F*T, C]"""

        window_attn = post_fuse_net[idx]
        merge = torch.cat((residual_q,dec), dim=-1)

        merge, Wh, Ww = window_attn(merge, Wh, Ww)
        return merge

    def csvq_layer(self, 
                   enc: torch.tensor, 
                   dec: torch.tensor, 
                   idx: int, 
                   vq: nn.Module, 
                   pre_fuse_net: nn.ModuleList = None, 
                   post_fuse_net: nn.ModuleList = None, 
                   Wh: int = None, 
                   Ww: int = None,
                   transmit: bool=True):
        # Quantization Forward that combines quantize and dequantize
        residual = self.pre_fuse(enc, dec, idx, pre_fuse_net, Wh, Ww)
        residual_q, cm_loss, cb_loss, kl_loss = vq(residual)
        dec_refine = self.post_fuse(residual_q, dec, idx, post_fuse_net, Wh, Ww, transmit)

        if not transmit:
            mask = torch.full((dec.shape[0],), fill_value=False, device=dec.device)
            cm_loss *= mask
            cb_loss *= mask
            kl_loss *= mask

        return dec_refine, cm_loss, cb_loss, kl_loss
    
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

        
        if self.fuse_net == "cross_w_merge":
            pre_fuse_net, post_fuse_net = self.init_w_attn_fuse_blocks(max_streams, h_dims=h_dims, f_dims=self.f_dims[1:], 
                                    window_size=window_size*2, fuse_attn_heads=swin_heads, shift_wa=False)
            
        elif self.fuse_net == "cross_sw_merge":
            pre_fuse_net, post_fuse_net = self.init_w_attn_fuse_blocks(max_streams, h_dims=h_dims, f_dims=self.f_dims[1:], 
                                    window_size=window_size*2, fuse_attn_heads=swin_heads, shift_wa=True)    
        
        elif self.fuse_net == "self_w_merge":
            pre_fuse_net, post_fuse_net = self.init_swin_merge_blocks(max_streams, h_dims=h_dims, window_size=window_size, fuse_attn_heads=swin_heads, 
                                        mlp_ratio=mlp_ratio, is_causal=is_causal, shift_wa=False)
            
        elif self.fuse_net == "self_sw_merge":
            pre_fuse_net, post_fuse_net = self.init_swin_merge_blocks(max_streams, h_dims=h_dims, window_size=window_size, fuse_attn_heads=swin_heads, 
                                        mlp_ratio=mlp_ratio, is_causal=is_causal, shift_wa=True)
        
        elif self.fuse_net == "None":
            pre_fuse_net = nn.ModuleList([nn.Identity() for _ in range(max_streams-1)])
            post_fuse_net = nn.ModuleList([nn.Identity() for _ in range(max_streams-1)])

            # pre_fuse_net = nn.ModuleList([
            #     nn.LayerNorm(h_dims[i]) for i in range(max_streams-1) 
            # ])
            # post_fuse_net = nn.ModuleList([
            #     nn.LayerNorm(h_dims[i]) for i in range(max_streams-1) 
            # ])

        else:
            raise ValueError("fuse_net method must be in [cross_w_merge, cross_sw_merge, self_w_merge, self_sw_merge]")

        self.pre_fuse_net, self.post_fuse_net = pre_fuse_net, post_fuse_net
        
        self.post_swin = SwinTransformerLayer(
                    self.out_h_dims[-1], self.out_h_dims[-1],
                    depth=swin_depth, num_heads=swin_heads[0],
                    window_size=window_size, mlp_ratio=mlp_ratio,
                    subsample=None, is_causal=is_causal
        )# tag

    def decode(self, enc_hs: list, streams: int, vqs: nn.ModuleList, Wh: int, Ww: int):
        """Step-wise Fuse decoding (Combines Quantize and Dequantize for Forward Training)
        Args: 
            enc_hs: a list of encoded features at multiple scale
            streams: number of bitstreams to use <= depth + 1
            vqs: a modulelist of quantizers with size $depth$
            Wh, Ww: encoder last feature size
        """
        assert streams <= self.max_streams and len(vqs) == self.max_streams

        if streams == 0:
            z0, cm_loss, cb_loss, kl_loss = enc_hs[-1], 0, 0, 0
        else:
            z0, cm_loss, cb_loss, kl_loss = vqs[0](enc_hs[-1])
        dec_hs = [z0]
        for i, blk in enumerate(self.blocks):
            transmit = (i < streams-1)    
            if self.training:
                dec_i_refine, cm_loss_i, cb_loss_i, kl_loss_i = self.csvq_layer(
                                                    enc=enc_hs[-1-i], dec=dec_hs[i],
                                                    idx=i, vq=vqs[i+1], 
                                                    pre_fuse_net=self.pre_fuse_net,
                                                    post_fuse_net=self.post_fuse_net,
                                                    Wh=Wh, Ww=Ww, transmit=transmit)
                cm_loss += cm_loss_i
                cb_loss += cb_loss_i
                kl_loss += kl_loss_i
            else:
                if transmit:
                    dec_i_refine, cm_loss_i, cb_loss_i, kl_loss_i = self.csvq_layer(
                                                            enc=enc_hs[-1-i], dec=dec_hs[i],
                                                            idx=i, vq=vqs[i+1], 
                                                            pre_fuse_net=self.pre_fuse_net,
                                                            post_fuse_net=self.post_fuse_net,
                                                            Wh=Wh, Ww=Ww)
                    cm_loss += cm_loss_i
                    cb_loss += cb_loss_i
                    kl_loss += kl_loss_i
                else:
                    dec_i_refine = dec_hs[i]
            
            dec_next, Wh, Ww = blk(dec_i_refine, Wh, Ww)
            dec_hs.append(dec_next)

        dec_next, Wh, Ww = self.post_swin(dec_next, Wh, Ww)
        recon_feat = self.patch_deembed(dec_next)

        return recon_feat, dec_hs, cm_loss, cb_loss, kl_loss
    
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
    
    def init_swin_merge_blocks(self, max_streams, window_size, h_dims, fuse_attn_heads:list, mlp_ratio, is_causal, shift_wa=False):

        if len(fuse_attn_heads) == 1:
            num_heads = fuse_attn_heads[0]
            num_heads = [min(num_heads*2**(len(self.h_dims)-i-1), num_heads*2**3) for i in range(len(self.h_dims))]
        else:
            num_heads = fuse_attn_heads[::-1]

        pre_fuse_net, post_fuse_net = [], []
        for i in range(max_streams-1):
            pre_block = SwinTransformerLayer(
                h_dims[i]*2,
                h_dims[i],
                depth=2 if shift_wa else 1,
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                subsample="down",
                scale_factor=[1,1],
                is_causal=is_causal
            )
            post_block = SwinTransformerLayer(
                h_dims[i]*2,
                h_dims[i],
                depth=2 if shift_wa else 1,
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                subsample="down",
                scale_factor=[1,1],
                is_causal=is_causal
            )
            pre_fuse_net.append(pre_block)
            post_fuse_net.append(post_block)

        return nn.ModuleList(pre_fuse_net), nn.ModuleList(post_fuse_net)

    def init_w_attn_fuse_blocks(self, max_streams, window_size, h_dims, f_dims, fuse_attn_heads: list, shift_wa=False):
        
        if len(fuse_attn_heads) == 1:
            num_heads = fuse_attn_heads[0]
            num_heads = [min(num_heads*2**(len(self.h_dims)-i-1), num_heads*2**3) for i in range(len(self.h_dims))]
        else:
            num_heads = fuse_attn_heads[::-1]

        shift_size = window_size//2
        pre_fuse_net, post_fuse_net = [], []
        for i in range(max_streams-1):
            if shift_wa:
                pre_block = nn.Sequential(*[
                    WindowAlignment(freq_size=f_dims[i], d_model=h_dims[i], 
                                    num_heads=num_heads[i], window_size=window_size, shift_size=0),
                    WindowAlignment(freq_size=f_dims[i], d_model=h_dims[i],
                                    num_heads=num_heads[i], window_size=window_size, shift_size=shift_size)
                    ]
                )
                post_block = nn.Sequential(*[
                    WindowAlignment(freq_size=f_dims[i], d_model=h_dims[i], 
                                    num_heads=num_heads[i], window_size=window_size, shift_size=0),
                    WindowAlignment(freq_size=f_dims[i], d_model=h_dims[i],
                                    num_heads=num_heads[i], window_size=window_size, shift_size=shift_size)
                ])
            else:
                pre_block = WindowAlignment(freq_size=f_dims[i], d_model=h_dims[i], 
                                            num_heads=num_heads[i], window_size=window_size, shift_size=0)
                post_block = WindowAlignment(freq_size=f_dims[i], d_model=h_dims[i], 
                                            num_heads=num_heads[i], window_size=window_size, shift_size=0)
            pre_fuse_net.append(pre_block)
            post_fuse_net.append(post_block)
        
        return nn.ModuleList(pre_fuse_net), nn.ModuleList(post_fuse_net)

    def vis_decoder(self):
        
        for i, blk in enumerate(self.blocks):

            print("Layer[{}]: swin_depth={} swin_hidden={} heads={} up={}".format(
                i, blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))

            if self.fuse_net in ["cross_w_merge", "cross_sw_merge"]:
                if i < len(self.pre_fuse_net):
                    fuse_mod = self.pre_fuse_net[i]
                    print("Cross Fuse Net: swin_hidden={} heads={}".format(
                    fuse_mod.attn.dim, fuse_mod.attn.num_heads
                ))
            elif self.fuse_net in ["self_w_merge", "self_sw_merge"]:
                if i < len(self.pre_fuse_net):
                    fuse_mod = self.pre_fuse_net[i]
                    print("Self Fuse Net: swin_depth={} swin_hidden={} heads={} up={}".format(
                    fuse_mod.depth, fuse_mod.swint_blocks[0].d_model, fuse_mod.swint_blocks[0].num_heads, blk.subsample!=None
                ))

        blk = self.post_swin
        print("Post-swin Layer: swin_depth={} swin_hidden={} heads={} up={}".format(
                blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))

if __name__ == "__main__":
    import os, yaml
    from utils import dict2namespace
    with open(os.path.join('src/configs', 'residual_9k_warmup.yml'), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    model = SwinAudioCodec(config.model.in_dim, config.model.in_freq, config.model.h_dims,
                               config.model.swin_depth, config.model.swin_heads, config.model.window_size, 
                               config.model.mlp_ratio, config.model.max_streams, config.model.overlap, 
                               config.model.num_vqs, config.model.proj_ratio, config.model.codebook_size, config.model.codebook_dims,
                               config.model.patch_size, config.model.use_ema, config.model.use_cosine_sim, 
                               config.model.is_causal, config.model.use_rvq, config.model.fuse_net, config.model.scalable, False,
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
    
    # codes, _ = model.encode(x=torch.ones(2, 47920), num_streams=6)
    # print(len(codes), codes[0].shape)

    # print(model.decode(codes, enc_feat_size=(1, 150)))
        
        
