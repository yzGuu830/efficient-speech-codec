# import sys
# sys.path.append('/Users/tracy/Desktop/Neural_Audio_Codec')
import torch.nn as nn
from models.modules.transformer import SwinTLayer
from models.modules.downsample import PatchDeEmbed, PatchSplit
import torch

import config as cfg

class SwinTDecoder(nn.Module):
    def __init__(self, init_H=192, in_channels=2, patch_size=(6,2), model_depth=4, layer_depth=2,
                 d_model=(16, 24, 32, 64), num_heads=(4, 4, 8, 16,), window_size=4, 
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, proj_drop=0., attn_drop=0., 
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 ) -> None:
        super().__init__()

        self.patch_size = patch_size

        self.decoder = nn.ModuleList() 
        d_model_out = (d_model[0],) + d_model
        for i in range(model_depth):
            subsample = None if i == model_depth-1 else PatchSplit

            self.decoder.append(
                        SwinTLayer( in_dim=d_model[model_depth-i-1],
                                    out_dim=d_model_out[model_depth-i-1],
                                    depth=layer_depth,
                                    num_heads=num_heads[model_depth-i-1],
                                    window_size=window_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    proj_drop=proj_drop,
                                    attn_drop=attn_drop,
                                    norm_layer=norm_layer,
                                    subsample=subsample,
                                    use_checkpoint=use_checkpoint)
            )
        self.deembed_layer = PatchDeEmbed(init_H=init_H,
                                      patch_size=patch_size,
                                      in_chans=in_channels,
                                      embed_dim=d_model[0],
                                      )

    def plain_decode(self, dec_in, ith, H, W):
        """Plain decoding without residual
        Args: 
            dec_in: ith decoded feature, tensor size (B, L_ith, d_model)
            ith: ith layer
        """
        dec_layer = self.decoder[ith]
        dec_in_next, H, W = dec_layer(dec_in, H, W)

        return dec_in_next, H, W

    def fuse_decode(self, dec_in, residual_q, ith, H, W):
        """Fuse decoding with residual fuse
        Args: 
            dec_in: ith decoded feature, tensor size (B, L_ith, d_model)
            enc_out: ith encoded feature to refine the process, tensor same size
            ith: ith layer
        """
        # print("dec_in", dec_in.shape)
        dec_in_refined = residual_q + dec_in
        dec_in_next, H, W = self.plain_decode(dec_in_refined, ith, H, W)

        return dec_in_next, H, W
    
    def quantize_residual(self, dec_in, enc_out, quantizer, H, W, verbose=False):
        residual = enc_out - dec_in 
        residual_code, L = quantizer.quantize(residual, H, W, verbose)

        return residual_code, L
    
    def dequantize_residual(self, residual_code, quantizer, H, W):
        residual_q = quantizer.dequantize(residual_code, H, W)

        return residual_q

    def forward(self, enc_hs, bitstreams, quantizers, H, W):
        """Step-wise Fuse decoding (Forward Training)
        Args: 
            enc_hs: a list of encoded features at multiple scale
            bitstreams: number of bitstreams to use <= depth + 1
            quantizers: a modulelist of quantizers with size $depth$
            H, W: lowest spec size
        """
        assert len(quantizers) == len(self.decoder) + 1, \
            f"quantizer number not right {len(quantizers)} {len(self.decoder)}"

        if bitstreams == 0:
            z, vq_loss = enc_hs[-1], torch.zeros(enc_hs[-1].size(0), device=enc_hs[-1].device) if cfg.num_workers > 1 \
                else torch.tensor(0.0, device=enc_hs[-1].device)
        else:
            z, vq_loss = quantizers[0](enc_hs[-1], H, W)

        dec_hs = [z]
        for i in range(len(self.decoder)):
            transmit = (i < bitstreams-1)

            dec_in, enc_out = dec_hs[i], enc_hs[-1-i]
            if transmit:
                residual_i = enc_out - dec_in
                residual_i_q, vq_loss_i = quantizers[i+1](residual_i, H, W)

                dec_next, H, W = self.fuse_decode(dec_in, residual_i_q, i, H, W)
                vq_loss += vq_loss_i

            else:
                dec_next, H, W = self.plain_decode(dec_in, i, H, W)

            dec_hs.append(dec_next)

        dec_out = self.deembed_layer(dec_hs[-1])    

        return dec_out, vq_loss, dec_hs
    
    def compress(self, enc_hs, bitstreams, quantizers, H, W, verbose=False):

        codes_0, size_0 = quantizers[0].quantize(enc_hs[-1], H, W, verbose)
        q_size, d_size = [size_0[0]], [size_0[1]]
        quantized_codes = [codes_0]

        if bitstreams < 2: 
            if verbose:
                print("Quantization Length: ", q_size)
                print("Quantization Total Dimensions: ", d_size)
            return quantized_codes, q_size

        z = quantizers[0].dequantize(codes_0, H, W)
        dec_hs = [z]
        for i in range(bitstreams-1):
            dec_in, enc_out, quantizer = dec_hs[i], enc_hs[-1-i], quantizers[i+1]

            residual_code_i, size_i = self.quantize_residual(dec_in, enc_out, quantizer, H, W, verbose) 
            if verbose: 
                q_size.append(size_i[0])
                d_size.append(size_i[1])
            quantized_codes.append(residual_code_i)

            if len(quantized_codes) == bitstreams: 
                if verbose:
                    print("Quantization Length: ", q_size)
                    print("Quantization Total Dimensions: ", d_size)
                return quantized_codes, q_size

            residual_i_q = self.dequantize_residual(residual_code_i, quantizer, H, W)
            dec_next, H, W = self.fuse_decode(dec_in, residual_i_q, i, H, W)
            dec_hs.append(dec_next)

    def decompress(self, bitstreams, quantizers, quantized_codes, H, W):
        assert len(quantized_codes) >= 1, "Codes not complete"
        z = quantizers[0].dequantize(quantized_codes[0], H, W)
        dec_hs = [z]

        for i in range(len(self.decoder)):
            transmit = (i < bitstreams-1)
            dec_in = dec_hs[i]

            if transmit:
                residual_code_i, quantizer = quantized_codes[i+1], quantizers[i+1]
                residual_i_q = self.dequantize_residual(residual_code_i, quantizer, H, W)

                dec_next, H, W = self.fuse_decode(dec_in, residual_i_q, i, H, W)

            else:
                dec_next, H, W = self.plain_decode(dec_in, i, H, W)

            dec_hs.append(dec_next)

        dec_out = self.deembed_layer(dec_hs[-1])
        return dec_out, dec_hs

    
if __name__ == "__main__":
    import torch
    from models.modules.vq import GroupQuantization
    enc_hs = [
        torch.randn(1, 128*300, 48),
        torch.randn(1, 64*300, 48*2**1),
        torch.randn(1, 32*300, 48*2**2),
        torch.randn(1, 16*300, 48*2**3),
        torch.randn(1, 8*300, 48*2**4),
        torch.randn(1, 4*300, 48*2**5),
    ]

    quantizers = nn.ModuleList([
            GroupQuantization(
                in_dim=(256//2**(6-j)) * 6*2**(6-j-1), 
                fix_dim=((256//2**(6-j)) * 6*2**(6-j-1))//2, 
                dim_map=True,
                num_overlaps=2, 
                num_groups=6, )
            for j in range(6)
    ])

    decoder = SwinTDecoder(d_model=6)

    dec_out, vq_loss, dec_hs = decoder(enc_hs, 4, quantizers, 4, 300)

    for hs in dec_hs:
        print(hs.shape)

    # codes = decoder.compress(enc_hs, 6, quantizers, H=4, W=300)
    # print(len(codes))
    # print(len(codes[0]))
    # print(codes[0][0].shape)

    # bits = 0
    # for bs in codes:
    #     for g in bs:
    #         bits += g.size(-1)
    # print("total bits: ", bits*10, "bps: ", bits*10/3)

    # dec_out, _ = decoder.decompress(6, quantizers, codes, H=4, W=300)

    # print(dec_out.shape)