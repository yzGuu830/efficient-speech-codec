import torch
import torch.nn as nn
from einops import rearrange

from .codebook import Codebook

class ProductVectorQuantize(nn.Module):
    "Product VQ Layer to Quantize Audio STFT features"
    def __init__(self, 
                in_dim: int,
                in_freq: int, 
                overlap: int=4,
                num_vqs: int=3, 
                codebook_dim: int=8,
                codebook_size: int=1024, 
                l2norm: bool=True,
                ) -> None: 
        super().__init__()

        self.overlap, self.codebook_dim, self.codebook_size, self.num_vqs = overlap, codebook_dim, codebook_size, num_vqs
        self.in_freq, self.in_dim, self.fix_dim = in_freq, in_dim, in_freq*in_dim # vector dimension after merging
        self.vq_dims = split_dimension(self.fix_dim*overlap, num_vqs) # vector dimension for each vq
        
        self.vqs = nn.ModuleList([ Codebook(codebook_dim, codebook_size, l2norm) for _ in self.vq_dims ]) 
        self.down_projs, self.up_projs = nn.ModuleList([
            nn.Linear(in_dim, codebook_dim, bias=False) for in_dim in self.vq_dims
            ]), nn.ModuleList([
                    nn.Linear(codebook_dim, in_dim, bias=False) for in_dim in self.vq_dims
                ])

    def forward(self, z_e, freeze_vq: bool=False):
        """ ProductVQ Forwrd Function Combining Encoding/Decoding
        Args: 
            z_e (Tensor): encoded vector with shape (B, H*W, C) [transformer output] or (B, C, H, W) [convolution output]
            freeze_vq (Boolean): apply codebook freezing during pre-training
        Returns:
            dict: 
                z_q: Tensor of quantized encoded vector with shape (B, H*W, C) [transformer output] or (B, C, H, W) [convolution output]
                codes: Tensor of indices with shape (B, group_size, T)
                cb_loss: codebook loss averaged
                cm_loss: commitment loss averaged
        """
        if not self.training and freeze_vq: 
            raise ValueError("``freeze_vq`` must be set False during inference")

        dims = len(z_e.shape)
        z_e = pre_process(z_e, self.in_freq, self.overlap, self.fix_dim, dims) # [B, W//overlap, overlap*H*C]
        z_q, codes = [], []

        s_idx, codebook_loss, commitment_loss = 0, 0., 0.
        for m, (down_proj, up_proj, vq) in enumerate(zip(self.down_projs, self.up_projs, self.vqs)):
            z_e_m = z_e[..., s_idx:s_idx+self.vq_dims[m]]
            
            z_e_m = down_proj(z_e_m)
            z_q_m, code, cb_loss, cm_loss = vq(z_e_m)
            if freeze_vq: # freeze codebook during pre-training phase
                z_q_m = z_q_m * 0. + z_e_m
                cb_loss *= 0.
                cm_loss *= 0.
            z_q_m = up_proj(z_q_m)

            codes.append(code)
            z_q.append(z_q_m)
            commitment_loss += cm_loss
            codebook_loss += cb_loss

            s_idx += self.vq_dims[m]
      
        return {"z_q": post_process(torch.cat(z_q, dim=-1), self.in_freq, self.overlap, self.fix_dim, dims),  # [B, H*W, C] / [B, C, H, W], 
                "codes": torch.stack(codes, dim=1),                                                           # [B, group_size, T] 
                "cb_loss": codebook_loss/self.num_vqs, 
                "cm_loss": commitment_loss/self.num_vqs}
    
    def encode(self, z_e):
        """ ProductVQ Encoding Process
        Args:
            z_e (Tensor): encoded vector with shape (B, H*W, C) [transformer output] or (B, C, H, W) [convolution output]
        Returns: 
            Tensor of indices with shape (B, group_size, T)
        """

        z_e = pre_process(z_e, self.in_freq, self.overlap, self.fix_dim, len(z_e.shape))
        
        s_idx, codes = 0, []
        for m, (down_proj, vq) in enumerate(zip(self.down_projs, self.vqs)):
            z_e_m = z_e[..., s_idx:s_idx+self.vq_dims[m]]
            code = self.vector_quantization_encode(z_e_m, down_proj, vq)
            codes.append(code)
            s_idx += self.vq_dims[m]

        return torch.stack(codes, dim=1)
    
    def decode(self, codes, dims=3):
        """ ProductVQ Decoding Process
        Args:
            codes (Tensor): Tensor of indices with shape (B, group_size, T)
            dims (int): 3 for transformer / 4 for convolution
        Returns: 
            Tensor of quantized encoded vector with shape (B, H*W, C) [transformer output] or (B, C, H, W) [convolution output]
        """
        z_q = []
        for m, (up_proj, vq) in enumerate(zip(self.up_projs, self.vqs)):
            code = codes[:, m:m+1, :]
            z_q_m  = self.vector_quantization_decode(code, up_proj, vq)
            z_q.append(z_q_m)

        z_q = post_process(torch.cat(z_q, dim=-1), self.in_freq, self.overlap, self.fix_dim, dims)
        return z_q
    
    def vector_quantization_encode(self, z_e_m: torch.tensor, down_proj: nn.Linear, vq: Codebook):
        """ Quantize sub-vector z_e_m into code
        Args: 
            z_e_m (Tensor): m-th encoded sub-vector in ProductVQ with shape (Bs, T, dim)
            down_proj (nn.Linear): down projection layer for latent dimension reduction 
            vq (Codebook): Codebook object for quantization
        Returns:
            Tensor of code with shape (Bs, T)
        """

        z_e_m = down_proj(z_e_m)
        code = vq.encode(z_e_m)
        return code
    
    def vector_quantization_decode(self, code: torch.tensor, up_proj: nn.Linear, vq: Codebook):
        """ Recover quantized sub-vector z_q_m from code
        Args: 
            code (Tensor): code for m-th encoded sub-vector in ProductVQ with shape (Bs, T)
            up_proj (nn.Linear): down projection layer for latent dimension reduction 
            vq (Codebook): Codebook object for quantization
        Returns:
            Tensor of quantized sub-vector z_q_m with shape (Bs, T, dim)
        """

        z_q_m = vq.decode(code)
        z_q_m = up_proj(z_q_m)
        return z_q_m


class ResidualVectorQuantize(nn.Module):
    "Residual VQ Layer"
    def __init__(self,
                in_dim: int=64,
                in_freq: int=6,
                hidden_dim: int=None,
                overlap: int=4,
                num_vqs: int=6, 
                codebook_dim: int=8,
                codebook_size: int=1024, 
                l2norm: bool=True,) -> None:
        super().__init__()

        self.overlap, self.codebook_dim, self.codebook_size, self.num_vqs = overlap, codebook_dim, codebook_size, num_vqs
        self.in_freq, self.in_dim, self.fix_dim = in_freq, in_dim, in_freq*in_dim # vector dimension after merging

        if hidden_dim is None: 
            hidden_dim = self.fix_dim*overlap

        self.do_proj = (hidden_dim != codebook_dim) 
        if self.do_proj: # project down only once (at bottleneck)
            self.proj_down = nn.Linear(hidden_dim, codebook_dim, bias=False)
            self.proj_up = nn.Linear(codebook_dim, hidden_dim, bias=False)

        self.vqs = nn.ModuleList([ 
                Codebook(codebook_dim, codebook_size, l2norm) for _ in range(num_vqs)
            ])
    
    def residual_vector_quantization(self, z_e, num_streams):
        """ Recursively Quantize Vector Residuals
        Args:
            z_e (Tensor): Tensor with shape (B, T, hidden)
            num_streams (int): Number of residual vqs used
        """
        # recursively quantize residuals
        z_q, indices = 0., []
        codebook_loss, commitment_loss = 0., 0.
        
        residual = z_e
        for i, vq in enumerate(self.vqs):
            if not self.training and i >= num_streams:
                break

            z_q_i, code, cb_loss, cm_loss = vq(residual)

            residual = residual - z_q_i
            if self.training and i >= num_streams:
                z_q_i = z_q_i * 0.
                cm_loss, cb_loss = cm_loss * 0., cb_loss * 0.

            z_q = z_q + z_q_i
            indices.append(code)

            commitment_loss += cm_loss
            codebook_loss += cb_loss
        
        indices = torch.stack(indices, dim=1) # [B, num_rvqs, T]
        return z_q, indices, commitment_loss, codebook_loss

    def forward(self, z_e, num_streams, freeze_vq: bool=False):
        """ ResidualVQ Forwrd Function.
        Args: 
            z_e (Tensor): Input vector with shape (B, H*W, C) [transformer output] or (B, C, H, W) [convolution output] (at bottleneck)
            num_streams (int): number of residual vqs used 
            freeze_vq (Boolean): apply codebook freezing during pre-training
        """
        dims = len(z_e.shape)
        z_e = pre_process(z_e, self.in_freq, self.overlap, self.fix_dim, dims) # [B, W//overlap, overlap*H*C]
        
        z_e = self.proj_down(z_e) if self.do_proj else z_e
        
        z_q, indices, commitment_loss, codebook_loss = self.residual_vector_quantization(z_e, num_streams)

        if freeze_vq: 
            z_q = z_e + z_q * 0.
            codebook_loss, commitment_loss = codebook_loss * 0., commitment_loss * 0.

        z_q = self.proj_up(z_q) if self.do_proj else z_q
    
        return {"z_q": post_process(z_q, self.in_freq, self.overlap, self.fix_dim, dims), # [B, H*W, C] / [B, C, H, W] 
                "codes": indices,                                                         # [B, num_rvqs, T] 
                "cb_loss": codebook_loss, 
                "cm_loss": commitment_loss}

    def quantize_to_code(self, z_e, num_streams):
        
        indices, residual = [], z_e
        for i, vq in enumerate(self.vqs):

            code_i = vq.encode(residual)
            indices.append(code_i)
            if len(indices) == num_streams:
                break
            
            z_q_i = vq.decode(code_i)
            residual = residual - z_q_i
            
        indices = torch.stack(indices, dim=1) # [B, num_streams, T]
        return indices

    def dequantize_code(self, codes):

        z_q = 0.
        for i in range(codes.size(1)):
            z_q += self.vqs[i].decode(codes[:, i])

        return z_q

    def encode(self, z_e, num_streams):
        """
        Args:
            z_e (Tensor): latent feature at bottleneck
            num_streams (int): number of residual vqs used
        Returns:
            Tensor of codes with shape (Bs, num_streams, T)
        """

        dims = len(z_e.shape)
        z_e = pre_process(z_e, self.in_freq, self.overlap, self.fix_dim, dims) # [B, W//overlap, overlap*H*C]
        z_e_down = self.proj_down(z_e) if self.do_proj else z_e    
        indices = self.quantize_to_code(z_e_down, num_streams)
        return indices

    def decode(self, codes, dims=3):
        """
        Args:
            codes (Tensor): codes of quantized residuals with shape (Bs, num_streams, T)
            dims (int): 3 for transformer / 4 for convolution
        Returns:
            Tensor of quantized latent feature at bottleneck
        """

        z_q_down = self.dequantize_code(codes)
        z_q = self.proj_up(z_q_down) if self.do_proj else z_q_down
        z_q = post_process(z_q, self.in_freq, self.overlap, self.fix_dim, dims) # [B, H*W, C] / [B, C, H, W]
        return z_q

class ProductResidualVectorQuantize(nn.Module):
    "Product Residual VQ Layer"
    def __init__(self,
                in_dim: int,
                in_freq: int, 
                overlap: int=4,
                num_pvqs: int=3,
                num_rvqs: int=6, 
                codebook_dim: int=8,
                codebook_size: int=1024, 
                l2norm: bool=True,) -> None:
        super().__init__()

        self.overlap, self.codebook_dim, self.codebook_size = overlap, codebook_dim, codebook_size
        self.in_freq, self.in_dim, self.fix_dim = in_freq, in_dim, in_freq*in_dim # vector dimension after merging
        self.vq_dims = split_dimension(self.fix_dim*overlap, num_pvqs)  # vector dimension for each vq

        self.vqs = nn.ModuleList([ 
                ResidualVectorQuantize(hidden_dim=dim, num_vqs=num_rvqs, codebook_dim=codebook_dim, 
                        codebook_size=codebook_size, l2norm=l2norm) for dim in self.vq_dims
                ]) 

    def forward(self, z_e, num_streams, freeze_vq: bool=False):
        """ ProductResidualVQ Forwrd Function.
        Args: 
            z_e (Tensor): Input vector with shape (B, H*W, C) [transformer output] or (B, C, H, W) [convolution output] (at bottleneck)
            num_streams (int): number of residual vqs used 
            freeze_vq (Boolean): apply codebook freezing during pre-training
        """

        dims = len(z_e.shape)
        z_e = pre_process(z_e, self.in_freq, self.overlap, self.fix_dim, dims)

        z_q, indices = [], []
        codebook_loss, commitment_loss = 0., 0.

        s_idx = 0
        for m, rvq in enumerate(self.vqs):
            z_e_m = z_e[..., s_idx:s_idx+self.vq_dims[m]]

            z_e_m = rvq.proj_down(z_e_m) if rvq.do_proj else z_e_m

            z_q_m, indices_m, cm_loss, cb_loss = rvq.residual_vector_quantization(z_e_m, num_streams)
            if freeze_vq: 
                z_q_m = z_e_m + z_q_m * 0.
                cm_loss, cb_loss = cm_loss * 0., cb_loss * 0.

            z_q_m = rvq.proj_up(z_q_m) if rvq.do_proj else z_q_m

            indices.append(indices_m) # [B, num_rvqs, T]
            z_q.append(z_q_m)
            commitment_loss += cm_loss
            codebook_loss += cb_loss

            s_idx += self.vq_dims[m]
    
        return {"z_q": post_process(torch.cat(z_q, dim=-1), self.in_freq, self.overlap, self.fix_dim, dims), # [B, H*W, C] / [B, C, H, W] 
                "codes": torch.stack(indices, dim=2),                                                        # [B, num_rvqs, num_pvqs, T]
                "cb_loss": codebook_loss/len(self.vqs), 
                "cm_loss": commitment_loss/len(self.vqs)}

    def encode(self, z_e, num_streams):
        """
        Args:
            z_e (Tensor): latent at bottleneck
            num_streams (int): number of recursive vqs used
        Returns:
            Tensor of codes with shape (Bs, num_streams, num_pvqs, T)
        """
        dims = len(z_e.shape)
        z_e = pre_process(z_e, self.in_freq, self.overlap, self.fix_dim, dims)

        indices = []
        s_idx = 0
        for m, rvq in enumerate(self.vqs):
            z_e_m = z_e[..., s_idx:s_idx+self.vq_dims[m]]

            z_e_m = rvq.proj_down(z_e_m) if rvq.do_proj else z_e_m
            indices_m = rvq.quantize_to_code(z_e_m, num_streams)
            
            indices.append(indices_m)
            s_idx += self.vq_dims[m]

        indices = torch.stack(indices, dim=2)
        return indices

    def decode(self, codes, dims=3):
        """
        Args:
            codes (Tensor): codes of quantized residuals with shape (B, num_streams, num_pvqs, T)
            dims (int): 3 for transformer / 4 for convolution
        Returns: 
            Tensor of quantized latent
        """

        z_q = []
        for m, rvq in enumerate(self.vqs):
            z_q_m = rvq.dequantize_code(codes[..., m, :])
            z_q_m = rvq.proj_up(z_q_m) if rvq.do_proj else z_q_m
            z_q.append(z_q_m)

        z_q = post_process(torch.cat(z_q, dim=-1), self.in_freq, self.overlap, self.fix_dim, dims) # [B, H*W, C] / [B, C, H, W]   
        return z_q

def split_dimension(total_dim, num):
    if total_dim % num == 0:
        dims = [total_dim//num for _ in range(num)]
    else:
        dims = [total_dim//num for _ in range(num-1)]
        dims += [total_dim - sum(dims)]
    return dims

def pre_process(z_e, in_freq, overlap, fix_dim, dims=3):
    """ Pre-process input vector (reshaping and overlapping)
    Args: 
        z_e (Tensor): Input vector with shape (B, H*W, C) [transformer output] or (B, C, H, W) [convolution output]
        in_freq (int): H
        overlap (int): Number of overlapped frames to quantize together
        fix_dim (int): C*H
        dims (int): 3 for transformer / 4 for convolution
    Returns: 
        Tensor of reshaped vector with shape (B, W//overlap, overlap*H*C)
    """
    if dims == 3: 
        z_e = rearrange(z_e, "b (h w) c -> b w (c h)", h=in_freq)
    elif dims == 4:
        z_e = rearrange(z_e, "b c h w -> b w (c h)")
    
    B, W = z_e.size(0), z_e.size(1)
    # overlap feature frames
    if overlap > 1:
        assert W % overlap == 0, "Time dimension must be multiple of overlap"
        z_e = z_e.view(B, W//overlap, overlap, fix_dim) \
            .reshape(B, W//overlap, overlap*fix_dim)    
    return z_e
    
def post_process(z_q, in_freq, overlap, fix_dim, dims=3):
    """ Post-process quantized vector
    Args: 
        z_q (Tensor): Quantized vector with shape (B, W//overlap, overlap*H*C)
        in_freq (int): H
        overlap (int): Number of overlapped frames to quantize together
        fix_dim (int): C*H
        dims (int): 3 for transformer / 4 for convolution
    Returns: 
        Tensor of recovered vector with shape (B, H*W, C) [transformer output] or (B, C, H, W) (convolution output)
    """
    # split overlapping frames
    if overlap > 1:
        z_q = z_q.view(z_q.size(0), -1, overlap, fix_dim) \
            .reshape(z_q.size(0), -1, fix_dim)
    if dims == 3: 
        z_q = rearrange(z_q, "b w (c h) -> b (h w) c", h=in_freq)
    elif dims == 4:
        z_q = rearrange(z_q, "b w (c h) -> b c h w", h=in_freq)

    return z_q