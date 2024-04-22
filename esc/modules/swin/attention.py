"""Window Attentions are adapted from from https://github.com/microsoft/Swin-Transformer"""
import torch
import torch.nn as nn
import numpy as np

from timm.models.layers import trunc_normal_, to_2tuple
from modules.swin.scale import PatchMerge, PatchSplit

class TransformerLayer(nn.Module):
    """ESC Building Transformer Layer"""
    def __init__(self, in_dim: int, out_dim: int, num_heads: int,
                 depth: int=2, window_size: int=4, mlp_ratio: float=2.,
                 qkv_bias: bool=True, qk_scale: float=None, proj_drop: float=0., attn_drop: float=0.,
                 activation=nn.GELU, norm_layer=nn.LayerNorm,
                 scale: str=None, scale_factor: tuple=(2,1)
                 ) -> None:
        super().__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        # Transformer Modules
        self.swint_blocks = nn.ModuleList([
            SwinBlock(
                 d_model=in_dim,
                 num_heads=num_heads, 
                 window_size=window_size, 
                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                 mlp_ratio=mlp_ratio, 
                 qkv_bias=qkv_bias, 
                 qk_scale=qk_scale, 
                 proj_drop=proj_drop, 
                 attn_drop=attn_drop, 
                 act_layer=activation, 
                 norm_layer=norm_layer,
                )
            for i in range(depth)]) # WA-Blocks combined with SWA-Blocks
        
        # Scaling Modules
        if scale is not None:
            scale_map = {"down": PatchMerge, "up": PatchSplit}
            assert scale in scale_map.keys(), "string scale must be down/up" 
            self.subsample = scale_map[scale](in_dim, out_dim, scale_factor, norm_layer)
        else:
            self.subsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            returns: x, H, W at next scale
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
            if isinstance(self.subsample, PatchMerge):
                ratio = self.subsample.scale_factor
                Wh, Ww = (H + 1) // ratio[0], (W + 1) // ratio[1] if ratio[1] > 1 else W
            elif isinstance(self.subsample, PatchSplit):
                ratio = self.subsample.scale_factor
                Wh, Ww = H * ratio[0], W * ratio[1]
            return x_scale, Wh, Ww
        else:
            return x, H, W

class SwinBlock(nn.Module):
    def __init__(self, 
                 d_model,
                 num_heads, 
                 window_size=4, 
                 shift_size=0,
                 mlp_ratio=2., 
                 mlp_out_dim=None,
                 qkv_bias=True, 
                 qk_scale=None, 
                 proj_drop=0., 
                 attn_drop=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
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
        mlp_out_dim = d_model if not mlp_out_dim else mlp_out_dim
        self.mlp = FeedForward(d_model, mlp_out_dim, mlp_hidden_dim, proj_drop, act_layer)

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

        x = x.contiguous().view(B, H*W, C)

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

