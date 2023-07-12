import torch.nn as nn
import torch

class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = in_dim
        self.down_map = nn.Linear(2 * in_dim, out_dim, bias=False)
        self.norm = norm_layer(2 * in_dim)

        self.scale_factor = (2, 1)

    def forward(self, x, H):
        """ Forward function downsample num_patches -> num_patches//2 (along frequency domain)
        Args:
            x: Input feature, tensor size (B, H*W, in_dim).
            H, W: num_patches along Freq and Time 
            returns downscaled feature, tensor size (B, H*W//2, out_dim).
        """
        B, num_patches, d_model = x.shape
        assert d_model == self.d_model, "input feature has wrong size"

        W = num_patches // H
        x = x.view(B, H, W, d_model)

        # padding
        pad_input = (H % 2 == 1)
        if pad_input:
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, H % 2))

        x0 = x[:, 0::2, :, :]  # B H/2 W C
        x1 = x[:, 1::2, :, :]  # B H/2 W C
        x = torch.cat([x0, x1], -1)  # B H/2 W 2*C

        x = x.reshape(B, -1, d_model*2)

        x = self.norm(x)
        x = self.down_map(x)                      # B num_patches//2 out_dim

        return x

class PatchSplit(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = in_dim
        self.up_map = nn.Linear(in_dim, out_dim*2, bias=False)
        self.norm = norm_layer(in_dim)
        
        self.scale_factor = (2, 1)

    def forward(self, x, H):
        """ Forward function upsample num_patches -> num_patches*2 (along frequency domain)
        Args:
            x: Input feature, tensor size (B, H*W, in_dim).
            H, W: num_patches along Freq and Time 
            returns upscaled feature, tensor size (B, H*W*2, out_dim).
        """
        B, num_patches, d_model = x.shape
        assert d_model == self.d_model, "input feature has wrong size"

        x = self.norm(x)
        x = self.up_map(x)                      # B, num_patches, out_dim*2

        W = num_patches // H
        x = x.view(B, H, W, -1)                 # B, H, W, out_dim*2

        # upsample
        x = pixel_shuffle(x, self.scale_factor) # B H*2, W, out_dim

        return x.reshape(B, num_patches*2, -1)

class PatchEmbed(nn.Module):
    """ 2D Patch Embedding """
    def __init__(self,
                 init_H=256,
                 patch_size=(6,2),
                 in_chans=2,
                 embed_dim=48,
                 norm_layer=None,
                 bias=True,):
        super().__init__()

        assert patch_size[1] <= 6, f"Current Patch_Size {patch_size} set too Large, may affect reconstruction performance"

        self.H_ = init_H // patch_size[0]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)                  # BCFT -> BCHW

        x = x.flatten(2).transpose(1, 2)  # BCHW -> BCL -> BLC
        
        x = self.norm(x)
        
        return x

class PatchDeEmbed(nn.Module):
    """ 2D Patch DeEmbedding """
    def __init__(self,
                 init_H=256,
                 patch_size=(6,2),
                 in_chans=2,
                 embed_dim=48,):
        super().__init__()

        self.patch_size = patch_size

        self.H_ = init_H // patch_size[0]

        self.de_proj1 = nn.Conv2d(embed_dim, embed_dim * patch_size[0] * patch_size[1], kernel_size=3, stride=1, padding=1)
                                       
        self.de_proj2 = nn.Conv2d(embed_dim, in_chans, kernel_size=5, stride=1, padding=2)   

    def forward(self, x):
        B, _, C = x.shape

        x = x.view(B, self.H_, -1, C) \
            .permute(0, 3, 1, 2).contiguous() # BCHW

        x = self.de_proj1(x)    # B C*up H W  
        x = pixel_shuffle(x.permute(0,2,3,1), self.patch_size)  # B C F=H*up W

        x = self.de_proj2(x.permute(0,3,1,2))     # BCHW -> BCFT -> B2FT

        return x

def pixel_unshuffle(input, downscale_factor):
    batch_size, in_height, in_width, in_channel = input.size()
    out_channel = in_channel * (downscale_factor[0] * downscale_factor[1])
    out_height = in_height // downscale_factor[0]
    out_width = in_width // downscale_factor[1]
    input_view = input.reshape(batch_size, out_height, downscale_factor[0], out_width, downscale_factor[1], in_channel)
    shuffle_out = input_view.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, out_height, out_width, out_channel)
    return shuffle_out

def pixel_shuffle(input, upscale_factor):
    batch_size, in_height, in_width, in_channel = input.size()
    out_channel = in_channel // (upscale_factor[0] * upscale_factor[1])
    out_height = in_height * upscale_factor[0]
    out_width = in_width * upscale_factor[1]
    input_view = input.reshape(batch_size, in_height, in_width, upscale_factor[0], upscale_factor[1], out_channel)
    shuffle_out = input_view.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, out_height, out_width, out_channel)
    return shuffle_out




