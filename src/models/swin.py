import torch
import torch.nn as nn
import numpy as np
import math

from timm.models.layers import trunc_normal_, to_2tuple


class SwinTLayer(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim,
                 depth,
                 num_heads,
                 window_size=4,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_drop=0.,
                 attn_drop=0.,
                 is_causal=False,
                 norm_layer=nn.LayerNorm,
                 subsample=None,
                 scale_factor=(2,1)
                 ) -> None:
        super().__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        self.swint_blocks = nn.ModuleList([
            SwinTBlock(
                 d_model=in_dim,
                 num_heads=num_heads, 
                 window_size=window_size, 
                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                 mlp_ratio=mlp_ratio, 
                 qkv_bias=qkv_bias, 
                 qk_scale=qk_scale, 
                 proj_drop=proj_drop, 
                 attn_drop=attn_drop, 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 causal=is_causal
                )
            for i in range(depth)])
        
        self.subsample = subsample(in_dim, out_dim, scale_factor, norm_layer) if subsample is not None else None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.swint_blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)

        if self.subsample is not None:
            x_scale = self.subsample(x, H)
            if isinstance(self.subsample, PatchMerging):
                ratio = self.subsample.scale_factor
                Wh, Ww = (H + 1) // ratio[0], (W + 1) // ratio[1] if ratio[1] > 1 else W
            elif isinstance(self.subsample, PatchSplit):
                ratio = self.subsample.scale_factor
                Wh, Ww = H * ratio[0], W * ratio[1]
            return x_scale, Wh, Ww
        else:
            return x, H, W


class SwinTBlock(nn.Module):
    def __init__(self, 
                 d_model,
                 num_heads, 
                 window_size=4, 
                 shift_size=0,
                 mlp_ratio=2., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 proj_drop=0., 
                 attn_drop=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 causal=False,
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(d_model)
        self.attn = WindowAttention(
            d_model, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)

        self.norm2 = norm_layer(d_model)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = FeedForward(d_model, d_model, mlp_hidden_dim, proj_drop, act_layer)

        self.causal = causal

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        if self.causal:
            num_window, N = x_windows.size(0), x_windows.size(1)
            causal_mask = torch.tril(torch.ones((N, N), device=x.device))
            causal_mask = causal_mask.unsqueeze(0).repeat(num_window, 1, 1)
            attn_mask = (causal_mask - 1) * 1e9

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = x.contiguous().view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class CrossWindowAttention(nn.Module):
    
    def __init__(self, 
                 dim, 
                 window_size, 
                 num_heads, 
                 qkv_bias=True) -> None:
        super().__init__()

        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q_matrix = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_matrix = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_matrix = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):

        """ Forward function.
        Args:
            q k v: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape

        q = self.q_matrix(q).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0,2,1,3).contiguous()
        k = self.k_matrix(k).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0,2,1,3).contiguous()
        v = self.v_matrix(v).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0,2,1,3).contiguous()
        # B_ num_heads N C//num_heads

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class WindowAlignment(nn.Module):

    """Proposed Cross Window Attention to align Encoder/Decoder Features"""

    def __init__(self, 
                 freq_size,
                 d_model,
                 num_heads, 
                 window_size=4, 
                 shift_size=0,
                 norm_layer=nn.LayerNorm) -> None:
        super().__init__()

        self.attn = CrossWindowAttention(dim=d_model, window_size=to_2tuple(window_size), num_heads=num_heads)
        self.shift_size = shift_size
        self.window_size = window_size

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if norm_layer is not None:
            self.norm1 = norm_layer(d_model)
            self.norm2 = norm_layer(d_model)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.H = freq_size

    def forward(self, feats):

        """
        x, y = feats
        x: attention key & attention value (like encoder features)
        y: attention query (like decoder features)
        Attend feature y to feature x
        """
        x, y = feats
        B, L, C = x.shape
        H = self.H
        W = L // H
        assert x.shape == y.shape

        x = self.norm1(x)
        y = self.norm2(y)
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        y = nn.functional.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift for feature y
        if self.shift_size > 0:
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_y = y

        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        y_windows = window_partition(shifted_y, self.window_size)
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(q=y_windows, k=x_windows, v=x_windows) # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x_aligned = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            y = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            y = shifted_y

        if pad_r > 0 or pad_b > 0:
            x_aligned = x_aligned[:, :H, :W, :]
            y = y[:, :H, :W, :]

        x_aligned = x_aligned.contiguous().view(B, H * W, C)
        y = y.contiguous().view(B, H * W, C)
        
        return x_aligned, y

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class FeedForward(nn.Module):
    """Transformer MLP Layer with dimension maps"""

    def __init__(self, in_dim, out_dim, d_ff=2048, dropout=0.1, act_layer=nn.GELU,):
        super().__init__() 
    
        self.linear_1 = nn.Linear(in_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, out_dim)

        self.act_layer = act_layer()
    
    def forward(self, x):
        x = self.act_layer(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim, scale_factor=(2,1), norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = in_dim
        self.down_map = nn.Linear(scale_factor[0] * scale_factor[1] * in_dim, out_dim, bias=False)
        self.norm = norm_layer(scale_factor[0] * scale_factor[1] * in_dim)

        self.scale_factor = scale_factor

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

        # x0 = x[:, 0::2, :, :]  # B H/2 W C
        # x1 = x[:, 1::2, :, :]  # B H/2 W C
        # x = torch.cat([x0, x1], -1)  # B H/2 W 2*C

        x = pixel_unshuffle(x, self.scale_factor)

        x = x.reshape(B, -1, d_model * self.scale_factor[0] * self.scale_factor[1])

        x = self.norm(x)
        x = self.down_map(x)                      # B num_patches//2 out_dim

        return x

class PatchSplit(nn.Module):
    def __init__(self, in_dim, out_dim, scale_factor=(2,1), norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = in_dim
        self.up_map = nn.Linear(in_dim, out_dim * scale_factor[0] * scale_factor[1], bias=False)
        self.norm = norm_layer(in_dim)
        
        self.scale_factor = scale_factor

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

        return x.reshape(B, num_patches*2*self.scale_factor[1], -1)

class PatchEmbed(nn.Module):
    """ 2D Patch Embedding """
    def __init__(self,
                 init_H=192,
                 patch_size=(3,2),
                 in_chans=2,
                 embed_dim=48,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        assert patch_size[1] <= 6, f"Current Patch_Size {patch_size} set too Large, may affect reconstruction performance"

        self.H_ = init_H // patch_size[0]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):

        x = self.proj(x)                  # BCFT -> BCHW

        x = x.flatten(2).transpose(1, 2)  # BCHW -> BCL -> BLC
        
        x = self.norm(x)
        
        return x

class PatchDeEmbed(nn.Module):
    """ 2D Patch DeEmbedding """
    def __init__(self,
                 init_H=192,
                 patch_size=(3,2),
                 in_chans=2,
                 embed_dim=48,):
        super().__init__()

        self.patch_size = patch_size

        self.H_ = init_H // patch_size[0]

        self.de_proj1 = nn.Identity() if patch_size[0]==1 and patch_size[1]==1 \
            else nn.Conv2d(embed_dim, embed_dim * patch_size[0] * patch_size[1], kernel_size=5, stride=1, padding=2)
                                       
        self.de_proj2 = nn.Conv2d(embed_dim, in_chans, kernel_size=3, stride=1, padding=1)   

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
