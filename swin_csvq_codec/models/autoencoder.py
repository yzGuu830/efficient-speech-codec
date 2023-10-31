import sys
sys.path.append("/Users/tracy/Desktop/Neural_Audio_Codec")
import warnings
warnings.filterwarnings("ignore")
from models.encoder import SwinTEncoder
from models.decoder import SwinTDecoder
from models.modules.vq import GroupQuantization
from models.modules.losses import MSE_LOSS, MEL_LOSS, MS_LOSS

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import math
import config as cfg


class Swin_Audio_Codec(nn.Module):

    def __init__(self, init_H=192, in_channels=2, patch_size=(6,2), model_depth=4, layer_depth=2,
                 d_model=(12, 24, 24, 36), num_heads=(3, 3, 6, 12), window_size=4, 
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, proj_drop=0., attn_drop=0., norm_layer=nn.LayerNorm,
                 vq_down_ratio=1.0, num_overlaps=(1, 1, 1, 1, 1,), num_groups=6, codebook_size=1024, vq_commit=1.0,
                 win_length=20, hop_length=5, sr=16e3, scalable=False,
                 ) -> None:
        super().__init__()
        self.init_H, self.patch_size = init_H, patch_size

        self.transformer_encoder = SwinTEncoder(
            init_H, in_channels, patch_size, model_depth, layer_depth,
            d_model, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, proj_drop, attn_drop, norm_layer
        )
        self.transformer_decoder = SwinTDecoder(
            init_H, in_channels, patch_size, model_depth, layer_depth,
            d_model, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, proj_drop, attn_drop, norm_layer
        )

        assert(d_model[0] % num_groups == 0), "Choose another dimension size"

        self.vqs = nn.ModuleList()
        q_dims = []
        for i in range(model_depth+1):
            in_dim = (init_H//patch_size[0] // 2**(model_depth-i)) * d_model[-i] if i > 0 else (init_H//patch_size[0] // 2**(model_depth-1)) * d_model[-1]
            # in_dim = d_model*2**(model_depth-i) if i >= 1 else d_model*2**(model_depth-1)
            dim_map = (vq_down_ratio != 1.0)
            fix_dim = int(in_dim * vq_down_ratio) if dim_map else in_dim

            self.vqs.append(
                GroupQuantization(
                    in_dim=in_dim, 
                    fix_dim=fix_dim, 
                    dim_map=dim_map,
                    num_overlaps=num_overlaps[i], 
                    num_groups=num_groups, 
                    codebook_size=codebook_size,
                    vq_commit=vq_commit
                    )
            )
            q_dims.append(fix_dim*num_overlaps[i])

        self.vis_model()
        print(f"SwinT Audio Codec Loaded: {self.max_bps}kbps in maximum with {model_depth} Symmetric Transformer Layers")

        self.scalable = scalable
        self.num_layers = model_depth
        
        self.ft = torchaudio.transforms.Spectrogram(n_fft=(init_H-1)*2, 
                                            win_length=int(win_length*sr*1e-3),
                                            hop_length=int(hop_length*sr*1e-3), power=None)
        self.ift = torchaudio.transforms.InverseSpectrogram(n_fft=(init_H-1)*2, 
                                            win_length=int(win_length*sr*1e-3),
                                            hop_length=int(hop_length*sr*1e-3))

        self.recon_loss = MSE_LOSS()
        self.mel_loss = MEL_LOSS()
        self.ms_loss = MS_LOSS()

    def vis_model(self):
        x = torch.randn(1, 2, cfg.init_H, 600)
        print("initial grid size: ", cfg.init_H//self.patch_size[0], 600//self.patch_size[1])

        enc_hs, h, w = self.transformer_encoder.encode(x)
        for hs in enc_hs:
            print("enc feat size: ", hs.shape)

        codes, q_dims = self.transformer_decoder.compress(enc_hs, cfg.num_streams, self.vqs, h, w, verbose=True)

        self.max_bps = 1e-3 * sum(q_dims) * 60 / 3
        print("Max Bitrate per Second is ", self.max_bps*1e3)
        return

    def forward_train(self, raw_feat, num_stream):
        """
        Args:
            raw_audio: bs, L
            raw_feat: bs, F, T, 2
            num_stream: bitstreams used
        """

        if self.scalable: # Scalable Training
            num_stream = np.random.randint(1,self.num_layers+2)

        x = raw_feat.permute(0,3,1,2).contiguous()
        enc_hs, h, w = self.transformer_encoder.encode(x)

        dec_out, vq_loss, dec_hs = self.transformer_decoder(enc_hs, num_stream, self.vqs, h, w)
        recon_feat = dec_out.permute(0,2,3,1).contiguous()

        # ms_loss = self.ms_loss(enc_hs, dec_hs, num_stream)
        ms_loss = torch.zeros(raw_feat.size(0), device=raw_feat.device) if cfg.num_workers > 1 else torch.tensor(0.0, device=raw_feat.device)

        recon_feat_complex = torch.view_as_complex(recon_feat) 
        recon_audio = self.ift(recon_feat_complex)

        return recon_audio, recon_feat, vq_loss, ms_loss
    
    def forward_test(self, raw_audio, raw_feat, audio_len, num_stream):

        if num_stream == 0: # Pretrain
            recon_audio, recon_feat, _, _ = self.forward_train(raw_feat, num_stream=0)

        else:
            codes, _ = self.compress(raw_audio, num_stream, raw_feat)
            recon_audio, recon_feat = self.decompress(codes,num_stream, audio_len=audio_len)

        return recon_audio, recon_feat
    
    def forward(self, input, audio_len, num_stream, train=True):
        raw_audio, raw_feat = input['audio'], input['feat']

        if train:
            recon_audio, recon_feat, vq_loss, ms_loss = self.forward_train(raw_feat, num_stream)

            recon_loss = self.recon_loss(recon_feat, raw_feat)

            mel_loss = self.mel_loss(raw_audio, recon_audio)
            # mel_loss = torch.zeros(raw_audio.size(0), device=raw_audio.device) if cfg.num_workers > 1 else torch.tensor(0.0, device=raw_audio.device)

            return {
                "loss": recon_loss + 0.25 * vq_loss + 0.25 * mel_loss + 0.25 * ms_loss,
                "recon_loss": recon_loss,
                "vq_loss": vq_loss,
                "raw_audio": raw_audio,
                "recon_audio": recon_audio,
                "raw_feat": raw_feat, 
                "recon_feat": recon_feat,
                "mel_loss": mel_loss,
                "ms_loss": ms_loss
            }
        
        else:
            recon_audio, recon_feat = self.forward_test(raw_audio, raw_feat, audio_len, num_stream)

            return {
                "raw_audio": raw_audio,
                "recon_audio": recon_audio,
                "raw_feat": raw_feat, 
                "recon_feat": recon_feat
            }
    
    def compress(self, raw_audio, num_stream, raw_feat=None, verbose=False):

        if raw_feat is None:
            raw_feat = torch.view_as_real(self.ft(raw_audio))

        x = raw_feat.permute(0,3,1,2).contiguous()
        enc_hs, h, w = self.transformer_encoder.encode(x)
        
        codes, q_dims = self.transformer_decoder.compress(enc_hs, num_stream, self.vqs, h, w, verbose)

        return codes, q_dims
    
    def decompress(self, codes, num_stream, audio_len=3):

        h = (self.init_H // self.patch_size[0]) // 2**(self.num_layers-1)
        w = audio_len*200 // self.patch_size[1]

        dec_out, _ = self.transformer_decoder.decompress(num_stream, self.vqs, codes, h, w)

        recon_feat = dec_out.permute(0,2,3,1).contiguous()
        recon_feat_complex = torch.view_as_complex(recon_feat) 
        recon_audio = self.ift(recon_feat_complex)

        return recon_audio, recon_feat
    

if __name__ == "__main__":
    import config as cfg 
    Model = Swin_Audio_Codec(
        init_H=cfg.init_H, in_channels=cfg.in_channels, patch_size=cfg.patch_size, 
        model_depth=cfg.model_depth, layer_depth=cfg.layer_depth,
        d_model=cfg.d_model, num_heads=cfg.num_heads, window_size=cfg.window_size, 
        mlp_ratio=cfg.mlp_ratio, qkv_bias=cfg.qkv_bias, qk_scale=cfg.qk_scale, 
        proj_drop=cfg.proj_drop, attn_drop=cfg.attn_drop, norm_layer=cfg.norm_layer,
        vq_down_ratio=cfg.vq_down_ratio, num_overlaps=cfg.num_overlaps, 
        num_groups=cfg.num_groups, codebook_size=cfg.codebook_size, vq_commit=cfg.vq_commit,
        win_length=cfg.win_length, hop_length=cfg.hop_length, sr=cfg.sr, scalable=cfg.scalable
    )

    # Model.vis_model()

    x = {"audio": torch.randn(1, 48000-80),
         "feat": torch.randn(1, cfg.init_H, 600, 2)}
    
    output = Model(x, 3, 5, True)

    # print(output['recon_feat'].shape)

    # print(output)