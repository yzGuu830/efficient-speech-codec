import torch
import torch.nn as nn
import sys
sys.path.append("/Users/tracy/Library/CloudStorage/GoogleDrive-cloudstorage.yuzhe@gmail.com/My Drive/Research/Audio_Signal_Coding/Deep-Audio-Signal-Coding/src")
from models.codec import BaseCodec, SwinEncoder, SwinDecoder



class ResidualCrossScaleCodec(BaseCodec):

    def __init__(self, 
                 in_dim: int=2, in_freq: int=192, h_dims: list=[45,72,96,144,192,384,768], max_streams: int=6, 
                 overlap: int=2, num_vqs: int=3, proj_ratio: float=1., 
                 codebook_size: int=1024, codebook_dims: list=[12], use_ema: bool=False, use_cosine_sim: bool=True, 
                 patch_size: list=[3,2], swin_depth: int = 2, swin_heads: list = [3,3,6,12,24,48],
                 window_size: int = 4, mlp_ratio: float = 4.,
                 mel_windows: int=[32,64,128,256,512,1024,2048], mel_bins: int=[5,10,20,40,80,160,320], 
                 win_len: int=20, hop_len: int=5, sr: int=16000, vq: str="GVQ",
                 vis: bool=False) -> None:
        super().__init__(in_dim, in_freq, h_dims, max_streams, overlap, num_vqs, proj_ratio, codebook_size, codebook_dims, patch_size, use_ema, use_cosine_sim, mel_windows, mel_bins, win_len, hop_len, sr, vq)

        self.scalable = True
        self.encoder = ReisdualCrossScaleEncoder(
            swin_depth, swin_heads, window_size, mlp_ratio,
            in_freq, patch_size, self.in_dim, self.enc_h_dims,
        )
        self.decoder = ResidualCrossScaleDecoder(
            swin_depth, swin_heads, window_size, mlp_ratio,
            in_freq, patch_size, self.in_dim, self.dec_h_dims,
        )
        self.quantizer = self.init_quantizer_icsvq(overlap, num_vqs, proj_ratio, codebook_size, codebook_dims, 
                            patch_size, use_ema, use_cosine_sim, vq)

        if vis:
            self.encoder.vis_encoder()
            self.decoder.vis_decoder()
            self.vis_quantization()

    def encode_to_latents(self, x: torch.tensor, n_streams: int, alpha: float=1.0):
        """ 
        Args: 
            x: stft feature, tensor of size (B, C=2, F, T)
            n_streams: number of streams to quantize (>=1)
            alpha: fade in (on n_streams th layer) for progressive training [0,1]
            returns: 
                latents: [q_0, q_1, q_2, ...] quantized residual latents ordered from shallow to deep
                cm_loss, cb_loss, kl_loss: codebook losses
                Wh, Ww: feature size after n_streams (always max_streams during training)
        """
        assert n_streams >= 1, "n_streams should be at least 1"
        feat, H, W = self.encoder.embed(x) # B, H*W, C

        Wh, Ww = H, W
        feat_q, cm_loss, cb_loss, kl_loss = 0, 0, 0, 0
        latents = []
        for i in range(self.max_streams):
            if self.training is False and i >= n_streams:
                break
            
            residual = feat - feat_q

            if self.training and i == (n_streams-1):
                residual *= alpha

            feat, Wh, Ww = self.encoder.down(residual, i, Wh, Ww)
            feat_q, cm_loss_i, cb_loss_i, kl_loss_i = self.quantizer[i](feat)

            if self.training and i == (n_streams-1):
                cm_loss *= alpha
                cb_loss *= alpha
                kl_loss *= alpha

            mask = (torch.full((x.shape[0],), fill_value=i, device=x.device) < n_streams)
            cm_loss += cm_loss_i*mask
            cb_loss += cb_loss_i*mask
            kl_loss += kl_loss_i*mask 
            latents.append(feat_q)


        return latents, cm_loss, cb_loss, kl_loss, (Wh, Ww)


    def decode_from_latents(self, latents: list, n_streams: int, Wh: int, Ww: int, alpha: float=1.0):
        """ 
        Args: 
            latents: list of quantized residuals, each of size (B, Wh*Ww, C)
            n_streams: number of streams to quantize (>=1)
            alpha: fade in (on n_streams th layer) for progressive training [0,1]
            Wh, Ww: quantized residual size of H and W for latents[-1]
            returns: embedded feature, 
        """
        assert n_streams >= 1, "n_streams should be at least 1"
        
        refine_feat = 0
        for i in range(self.max_streams):
            if self.training is False:
                if self.max_streams-i <= n_streams: # latents has n_streams length during testing
                    q_residual = latents[self.max_streams-i-1]
                    refine_feat += q_residual
                    refine_feat, Wh, Ww = self.decoder.up(refine_feat, i, Wh, Ww)

            else: # latents has max_streams length during training
                if i == (self.max_streams-n_streams):
                    refine_feat *= alpha
                q_residual = latents[len(latents)-1-i]
                refine_feat = refine_feat + q_residual
                refine_feat, Wh, Ww = self.decoder.up(refine_feat, i, Wh, Ww)

                mask = (torch.full((q_residual.shape[0],), fill_value=(self.max_streams-i), device=q_residual.device) <= n_streams)
                refine_feat *= mask[:, None, None]
        
        recon_x = self.decoder.deembed(refine_feat, Wh, Ww)
        return recon_x
        

    def train_one_step(self, x, x_feat=None, streams=6, alpha=1.0):
        self.train()
        if x_feat is None:
            x_feat = self.spec_transform(x)
        else:
            x_feat = x_feat.permute(0,3,1,2)

        latents, cm_loss, cb_loss, kl_loss, (Wh, Ww) = self.encode_to_latents(x_feat, n_streams=streams, alpha=alpha)
        recon_feat = self.decode_from_latents(latents, n_streams=streams, Wh=Wh, Ww=Ww, alpha=alpha)
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

        latents, cm_loss, cb_loss, kl_loss, (Wh, Ww) = self.encode_to_latents(x_feat, n_streams=streams)
        recon_feat = self.decode_from_latents(latents, n_streams=streams, Wh=Wh, Ww=Ww)
        rec_x = self.audio_reconstruct(recon_feat)

        return {
                "raw_audio": x,
                "recon_audio": rec_x,
                "raw_feat": x_feat, 
                "recon_feat": recon_feat,
            }
    
    def forward(self, x, x_feat, streams, train=False, alpha=1.0):
        if train:
            return self.train_one_step(x, x_feat, streams, alpha)
        else:
            return self.test_one_step(x, x_feat, streams)
        
    # @torch.inference_mode()
    # def encode(self, x, num_streams=6):
    #     self.eval()

    #     x_feat = self.spec_transform(x)
    #     enc_hs, Wh, Ww = self.encoder.encode(x_feat)

    #     multi_codes = self.decoder.quantize(enc_hs, num_streams, vqs=self.quantizer, Wh=Wh, Ww=Ww)
    #     return multi_codes, (Wh, Ww)
    
    # @torch.inference_mode()
    # def decode(self, multi_codes, enc_feat_size=(2, 300)):
    #     self.eval()

    #     Wh, Ww = enc_feat_size
    #     dec_hs = self.decoder.dequantize(multi_codes, vqs=self.quantizer, Wh=Wh, Ww=Ww)

    #     rec_feat = dec_hs[-1]
    #     rec_x = self.audio_reconstruct(rec_feat)
    #     return rec_x

class ReisdualCrossScaleEncoder(SwinEncoder):
    def __init__(self, 
                 swin_depth: int = 2, swin_heads: list = [3,3,6,12,24], 
                 window_size: int = 4, mlp_ratio: float = 4, 
                 in_freq: int = 192, patch_size: list = [3,2], 
                 in_dim: int = 2, h_dims: list = [45,45,72,96,192,384], is_causal: bool = False) -> None:
        super().__init__(swin_depth, swin_heads, window_size, mlp_ratio, in_freq, patch_size, in_dim, h_dims, is_causal)


    def down(self, x: torch.tensor, idx: int, Wh: int, Ww: int):
        """ one layer forward (downsample the input feature x)
        Args: 
            idx: index for the encoder layer
            Wh, Ww: input feature's patchH and patchW
            returns: downscaled feature x, downscaled patchH and patchW
        """
        layer = self.blocks[idx]  
        x, Wh, Ww = layer(x, Wh, Ww)      

        return x, Wh, Ww

    def embed(self, x):
        """ patch embed & pre transformer layer
        Args: 
            x: stft feature, tensor of size (B, C=2, F, T)
            returns: embedded feature, 
        """
        Wh, Ww = x.size(2) // self.patch_size[0], x.size(3) // self.patch_size[1]
        x = self.patch_embed(x) # B, H, W, C

        x, Wh, Ww = self.pre_swin(x, Wh, Ww)
        return x, Wh, Ww

    def vis_encoder(self):
        blk = self.pre_swin
        print("Pre-swin Layer: swin_depth={} swin_hidden={} heads={} down={}".format(
                blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))
        for i, blk in enumerate(self.blocks):
            print("Layer[{}]: swin_depth={} swin_hidden={} heads={} down={}".format(
                i, blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))


class ResidualCrossScaleDecoder(SwinDecoder):
    def __init__(self, 
                 swin_depth: int = 2, swin_heads: list = ..., 
                 window_size: int = 4, mlp_ratio: float = 4, 
                 in_freq: int = 192, patch_size: list = ..., 
                 in_dim: int = 2, h_dims: list = ..., is_causal: bool = False) -> None:
        super().__init__(swin_depth, swin_heads, window_size, mlp_ratio, in_freq, patch_size, in_dim, h_dims, is_causal)

    def up(self, x: torch.tensor, idx: int, Wh: int, Ww: int):
        """ one layer forward (upsample the input feature x)
        Args: 
            idx: index for the decoder layer
            Wh, Ww: input feature's patchH and patchW
            returns: upscaled feature x, downscaled patchH and patchW
        """
        layer = self.blocks[idx]  
        x, Wh, Ww = layer(x, Wh, Ww)      

        return x, Wh, Ww

    def deembed(self, dec: torch.tensor, Wh: int, Ww: int):
        """ post transformer layer & patch embed
        Args: 
            dec: last decoded feature, tensor of size (B, H*W, C)
            Wh, Ww: input feature's patchH and patchW
            returns: x_hat (reconstructed stft feature), tensor of size (B, C=2, F, T)
        """

        dec_next, Wh, Ww = self.post_swin(dec, Wh, Ww)
        recon_feat = self.patch_deembed(dec_next)
        return recon_feat
    
    def vis_decoder(self):
        for i, blk in enumerate(self.blocks):
            print("Layer[{}]: swin_depth={} swin_hidden={} heads={} up={}".format(
                i, blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))
        blk = self.post_swin
        print("Post-swin Layer: swin_depth={} swin_hidden={} heads={} up={}".format(
                blk.depth, blk.swint_blocks[0].d_model, blk.swint_blocks[0].num_heads, blk.subsample!=None
            ))



if __name__ == "__main__":
    import os, yaml
    from utils import dict2namespace
    with open(os.path.join('configs', 'residual_9k_res_csvq.yml'), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    model = ResidualCrossScaleCodec(config.model.in_dim, config.model.in_freq, config.model.h_dims, 
                 config.model.max_streams, 
                 config.model.overlap, config.model.num_vqs, config.model.proj_ratio,
                 config.model.codebook_size, config.model.codebook_dims, config.model.use_ema, config.model.use_cosine_sim,
                 config.model.patch_size, config.model.swin_depth, config.model.swin_heads,
                 config.model.window_size, config.model.mlp_ratio,
                 config.model.mel_windows, config.model.mel_bins,
                 config.model.win_len, config.model.hop_len, config.model.sr, config.model.vq, True)

    x = torch.randn(9, 2, 192, 600)

    trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
    )
    print(trainable_params)

    latents, cm_loss, cb_loss, kl_loss, (Wh, Ww) = model.encode_to_latents(x, n_streams=6)
    for l in latents:
        print(l.shape)
    print(cm_loss, cb_loss, kl_loss, Wh, Ww)

    recon_feat = model.decode_from_latents(latents, n_streams=6, Wh=Wh, Ww=Ww)
    print(recon_feat.shape)
    # model.train()  
    # latents, cm_loss, cb_loss, kl_loss, (Wh, Ww) = model.encode_to_latents(x, n_streams=5)
    # for l in latents:
    #     print(l.shape)

    # recon_feat = model.decode_from_latents(latents, n_streams=5, Wh=Wh, Ww=Ww)
    # print(recon_feat.shape)