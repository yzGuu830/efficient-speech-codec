from einops import rearrange
import torch.nn as nn

class PatchMerging(nn.Module):
    """Module to Compress Vector"""
    def __init__(self,
                 in_dim: int, 
                 out_dim: int, 
                 scale_factor: tuple=(2,1), 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = in_dim
        self.down = nn.Linear(scale_factor[0]*scale_factor[1]*in_dim, 
                              out_dim if out_dim else scale_factor[0]*scale_factor[1]*in_dim*2, 
                              bias=False)
        self.norm = norm_layer(scale_factor[0]*scale_factor[1]*in_dim)
        self.scale_factor = scale_factor

    def forward(self, x, H):
        """ Forward function downsample num_patches -> num_patches//2 (along frequency domain)
        Args:
            x: Input feature, tensor size (B, H*W, in_dim).
            H, W: num_patches along Freq and Time 
            returns downscaled feature, tensor size (B, H*W//2, out_dim).
        """
        if sum(self.scale_factor) > 2:
            x = rearrange(x, "b (h w) c -> b h w c", h=H)

            pad_input = (H % 2 == 1)
            if pad_input:
                x = nn.functional.pad(x, (0, 0, 0, 0, 0, H % 2))

            x = pixel_unshuffle(x, self.scale_factor)
            x = rearrange(x, "b h w c -> b (h w) c")

        x = self.norm(x)
        x = self.down(x)
        return x

class PatchSplit(nn.Module):
    """Module to Reconstruct Vector"""
    def __init__(self,
                 in_dim: int, 
                 out_dim: int, 
                 scale_factor: tuple=(2,1), 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = in_dim
        self.up = nn.Linear(in_dim, 
                            out_dim*scale_factor[0]*scale_factor[1], 
                            bias=False)
        self.norm = norm_layer(in_dim)
        self.scale_factor = scale_factor

    def forward(self, x, H):
        """ Forward function upsample num_patches -> num_patches*2 (along frequency domain)
        Args:
            x: Input feature, tensor size (B, H*W, in_dim).
            H, W: num_patches along Freq and Time 
            returns upscaled feature, tensor size (B, H*W*2, out_dim).
        """

        x = self.norm(x)
        x = self.up(x)                    

        if sum(self.scale_factor) > 2:
            x = rearrange(x, "b (h w) c -> b h w c", h=H)
            x = pixel_shuffle(x, self.scale_factor)
            x = rearrange(x, "b h w c -> b (h w) c")
        return x



class PatchEmbed(nn.Module):
    """ 2D Patch Embedding """
    def __init__(self,
                 init_H: int=192,
                 patch_size: tuple=(3,2),
                 in_chans: int=2,
                 embed_dim: int=48,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        assert patch_size[1] <= 6, f"Current Patch_Size {patch_size} set too Large, may affect reconstruction performance"

        self.H_ = init_H // patch_size[0]
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                              kernel_size=patch_size, 
                              stride=patch_size)
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):

        x = self.proj(x)                          # BCFT -> BCHW
        x = rearrange(x, "b c h w -> b (h w) c")  # BCHW -> BCL -> BLC
        x = self.norm(x)
        return x
    

class PatchDeEmbed(nn.Module):
    """ 2D Patch DeEmbedding """
    def __init__(self,
                 init_H: int=192,
                 patch_size: tuple=(3,2),
                 in_chans: int=2,
                 embed_dim: int=48,):
        super().__init__()

        self.patch_size = patch_size
        self.H_ = init_H // patch_size[0]

        self.de_proj1 = nn.Identity() if patch_size[0]==1 and patch_size[1]==1 \
            else nn.Conv2d(embed_dim, # pixel shuffle
                           embed_dim * patch_size[0] * patch_size[1], 
                           kernel_size=5, stride=1, padding=2)
        self.de_proj2 = nn.Conv2d(embed_dim, 
                                  in_chans, # linear deembedding
                                  kernel_size=3, stride=1, padding=1)   

    def forward(self, x):

        x = rearrange(x, "b (h w) c -> b c h w", h=self.H_)
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