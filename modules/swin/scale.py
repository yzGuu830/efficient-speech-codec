from einops import rearrange
import torch.nn as nn

def pixel_unshuffle(input, downscale_factor:tuple=(2,1)):
    s1, s2 = downscale_factor
    B, H, W, C = input.size()
    C_, H_, W_ = C*(s1*s2), H//s1, W//s2

    unshuffle_out = input.reshape(B, H_, s1, W_, s2, C).\
        permute(0,1,3,2,4,5).reshape(B, H_, W_, C_)
    return unshuffle_out

def pixel_shuffle(input, upscale_factor:tuple=(2,1)):
    s1, s2 = upscale_factor
    B, H, W, C = input.size()
    C_, H_, W_ = C//(s1*s2), H*s1, W*s2

    shuffle_out = input.reshape(B, H, W, s1, s2, C_).\
        permute(0,1,3,2,4,5).reshape(B, H_, W_, C_)
    return shuffle_out


class PatchEmbed(nn.Module):
    """ 2D Linear Patchify """
    def __init__(self,
                 freq: int=192,
                 in_chans: int=2,
                 patch_size: tuple=(3,2),
                 embed_dim: int=48,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.H = freq // patch_size[0]
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):

        x = self.proj(x)                          # B2FT -> BCHW
        x = rearrange(x, "b c h w -> b (h w) c")  # BCHW -> BCL -> BLC
        x = self.norm(x)
        return x
    
class PatchDeEmbed(nn.Module):
    """ 2D Linear De-Patchify """
    def __init__(self,
                 freq: int=192,
                 in_chans: int=2,
                 patch_size: tuple=(3,2),
                 embed_dim: int=48,):
        super().__init__()

        self.patch_size = patch_size
        self.H = freq // patch_size[0]

        self.de_proj1 = nn.Conv2d(embed_dim,
                           embed_dim*patch_size[0]*patch_size[1], 
                           kernel_size=5, stride=1, padding=2)
        self.de_proj2 = nn.Conv2d(embed_dim, 
                                  in_chans,
                                  kernel_size=3, stride=1, padding=1)   

    def forward(self, x):
        x = rearrange(x, "b (h w) c -> b c h w", h=self.H)
        x = self.de_proj1(x)                                    # B C*scale H W  
        x = pixel_shuffle(x.permute(0,2,3,1), self.patch_size)  # B F T C
        x = self.de_proj2(x.permute(0,3,1,2))                   # BCFT -> B2FT
        return x

class PatchMerge(nn.Module):
    """Patch Merging Layer: Perform Pixel Unshuffle and Downscale"""
    def __init__(self,
                 in_dim: int, 
                 out_dim: int, 
                 scale_factor: tuple=(2,1), 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        s1, s2 = scale_factor

        self.norm = norm_layer(s1*s2*in_dim)
        self.down = nn.Linear(s1*s2*in_dim, out_dim, bias=False)
        self.scale_factor = scale_factor

    def forward(self, x, H):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, in_dim)
            H: num_patches along Freq Domain 
            returns: downscaled feature x, tensor size (B, H*W//2, out_dim)
        """
        
        x = rearrange(x, "b (h w) c -> b h w c", h=H)
        pad_input = (H%2 == 1)
        if pad_input:
            x = nn.functional.pad(x, (0,0,0,0,0,H%2))

        x = pixel_unshuffle(x, self.scale_factor)
        x = rearrange(x, "b h w c -> b (h w) c")
        x = self.norm(x)
        x = self.down(x)

        return x

class PatchSplit(nn.Module):
    """Patch Splitting Layer: Perform Pixel Shuffle and Upscale"""
    def __init__(self,
                 in_dim: int, 
                 out_dim: int, 
                 scale_factor: tuple=(2,1), 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        s1, s2 = scale_factor

        self.norm = norm_layer(in_dim)
        self.up = nn.Linear(in_dim, out_dim*s1*s2, bias=False)
        self.scale_factor = scale_factor

    def forward(self, x, H):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, in_dim)
            H: num_patches along Freq Domain 
            returns: upscaled feature x, tensor size (B, H*W*2, out_dim)
        """

        x = self.norm(x)
        x = self.up(x)                    

        x = rearrange(x, "b (h w) c -> b h w c", h=H)
        x = pixel_shuffle(x, self.scale_factor)
        x = rearrange(x, "b h w c -> b (h w) c")
        return x
    