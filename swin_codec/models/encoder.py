import sys
sys.path.append('/Users/tracy/Desktop/Neural_Audio_Codec')
import torch.nn as nn
from models.modules.transformer import SwinTLayer
from models.modules.downsample import PatchEmbed, PatchMerging

class SwinTEncoder(nn.Module):
    def __init__(self, init_H=192, in_channels=2, patch_size=(6,2), model_depth=4, layer_depth=2,
                 d_model=(16, 24, 32, 64), num_heads=(4, 4, 8, 16,), window_size=4, 
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, proj_drop=0., attn_drop=0., 
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 ) -> None:
        super().__init__()

        self.patch_size = patch_size

        self.embed_layer = PatchEmbed(init_H=init_H,
                                      patch_size=patch_size,
                                      in_chans=in_channels,
                                      embed_dim=d_model[0],
                                      )

        self.encoder = nn.ModuleList() 

        d_model_in = (d_model[0],) + d_model
        for i in range(model_depth):
            subsample = None if i == 0 else PatchMerging
            self.encoder.append(
                        SwinTLayer( in_dim=d_model_in[i],
                                    out_dim=d_model[i],
                                    depth=layer_depth,
                                    num_heads=num_heads[i],
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
        
    def encode(self, x):
        """Step-wise Encoding with downscaling
        Args: 
            x: spectrogram feature, tensor size (B, C=2, F, T)
            returns: encoder hidden states at multiple scale, eventual img size
        """

        Wh, Ww = self.embed_layer.H_, x.size(-1) // self.patch_size[1]
        x = self.embed_layer(x) # B, H*W, d_model
        # print(x.shape, Wh, Ww)
        enc_hs = [x]
        for layer in self.encoder:
            
            x, Wh, Ww = layer(x, Wh, Ww)
            enc_hs.append(x)

            # print(x.shape, Wh, Ww)

        return enc_hs, Wh, Ww
    
if __name__ == "__main__":
    import torch
    import config as cfg
    x = torch.randn(1, 2, cfg.init_H, 600)

    encoder = SwinTEncoder()

    enc_hs, H, W = encoder.encode(x)

    for hs in enc_hs:
        print(hs.shape)