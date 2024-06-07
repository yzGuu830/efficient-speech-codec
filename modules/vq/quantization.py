import torch
import torch.nn as nn
from einops import rearrange

from modules.vq.codebook import Codebook
from modules.vq.initialize import codebook_init_forward_hook_pvq


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
        self.in_freq, self.in_dim = in_freq, in_dim
        
        self.fix_dim = in_freq*in_dim # dimension after reshaping
        self.vq_dims = split_dimension(self.fix_dim*overlap, num_pvqs)

        self.vqs = nn.ModuleList([ 
                # Codebook(in_dim, codebook_dim, codebook_size, l2norm) for in_dim in self.vq_dims,
                ResidualVectorQuantize(hidden_dim=dim, num_vqs=num_rvqs, codebook_dim=codebook_dim, 
                        codebook_size=codebook_size, l2norm=l2norm) for dim in self.vq_dims
                ]) 

    def forward(self, z_e, num_streams, freeze=False):
        """ Product Residual VQ Forwrd Function (perform RVQ in each group)
        Args: 
            z_e: Input vector with shape (B, H*W, C) [swinT output] or (B, C, H, W) [conv output] (at bottleneck)
            num_streams: number of residual vqs used 
            freeze: boolean (True for handling pre-training stage, when codebook is not updated)
        """
        if freeze: num_streams = 0

        dims = len(z_e.shape)
        z_e = pre_process(z_e, self.in_freq, self.overlap, self.fix_dim, dims)

        z_q, indices = [], []
        codebook_loss, commitment_loss = 0., 0.

        s_idx = 0
        for i, rvq in enumerate(self.vqs):
            e_idx = s_idx+self.vq_dims[i]
            z_e_i = z_e[..., s_idx:e_idx]

            z_e_down_i = rvq.proj_down(z_e_i) if rvq.do_proj else z_e_i

            z_q_down_i, indices_i, cm_loss, cb_loss = rvq.residual_vector_quantize(z_e_down_i, num_streams)

            if freeze: z_q_down_i = z_e_down_i + z_q_down_i*0.

            z_q_i = rvq.proj_up(z_q_down_i) if rvq.do_proj else z_q_down_i

            indices.append(indices_i) # [B, num_rvqs, T]
            z_q.append(z_q_i)

            commitment_loss += cm_loss
            codebook_loss += cb_loss
            s_idx = e_idx

        z_q = post_process(torch.cat(z_q, dim=-1), self.in_freq, self.overlap, self.fix_dim, dims) # [B, H*W, C] / [B, C, H, W]      
        indices = torch.stack(indices, dim=2)                                                      # [B, num_rvqs, num_pvqs, T]

        return (z_q, indices), (commitment_loss/len(self.vqs), codebook_loss/len(self.vqs))

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
        self.in_freq, self.in_dim = in_freq, in_dim

        self.fix_dim = in_freq*in_dim # dimension after reshaping

        if hidden_dim is None: hidden_dim = self.fix_dim*overlap

        self.do_proj = (hidden_dim != codebook_dim) 
        if self.do_proj: # project down only once (at bottleneck)
            self.proj_down = nn.Linear(hidden_dim, codebook_dim, bias=False)
            self.proj_up = nn.Linear(codebook_dim, hidden_dim, bias=False)

        self.vqs = nn.ModuleList([ 
                Codebook(codebook_dim, codebook_dim, codebook_size, l2norm) for _ in range(num_vqs)
                # perform no projection in each codebook
            ])
        
        for i in range(num_vqs):
            nn.init.kaiming_normal_(self.vqs[i].embedding.weight) 	
        # print("Initializing Residual VQs with KaimingNormal")
    
    def residual_vector_quantize(self, z_e, num_streams):
        """ Recursively Quantize Vector Residuals
        Args:
            z_e: Tensor with shape (B, T, hidden)
            num_streams: Number of residual vqs used
        """
        # recursively quantize residuals
        z_q, indices = 0., []
        codebook_loss, commitment_loss = 0., 0.
        
        residual = z_e
        for i, vq in enumerate(self.vqs):
            if not self.training and i >= num_streams:
                break

            outputs = vq(residual, False)
            z_q_i, _, code_i = outputs["z_q"], outputs["z_e_down"], outputs["code"]
            cm_loss, cb_loss = outputs["commitment_loss"], outputs["codebook_loss"]

            residual = residual - z_q_i
            if self.training and i >= num_streams:
                z_q_i = z_q_i*0.
                cm_loss, cb_loss = cm_loss*0., cb_loss*0.

            z_q = z_q + z_q_i
            indices.append(code_i)

            commitment_loss += cm_loss
            codebook_loss += cb_loss
        
        indices = torch.stack(indices, dim=1) # [B, num_rvqs, T]
        return z_q, indices, commitment_loss, codebook_loss

    def forward(self, z_e, num_streams, freeze=False):
        """ Residual VQ Forwrd Function.
        Args: 
            z_e: Input vector with shape (B, H*W, C) [swinT output] or (B, C, H, W) [conv output] (at bottleneck)
            num_streams: number of residual vqs used 
            freeze: boolean (True for handling pre-training stage, when codebook is not updated)
        """
        if freeze: num_streams = 0

        dims = len(z_e.shape)
        z_e = pre_process(z_e, self.in_freq, self.overlap, self.fix_dim, dims) # [B, W//overlap, overlap*H*C]
        
        z_e_down = self.proj_down(z_e) if self.do_proj else z_e
        
        z_q_down, indices, commitment_loss, codebook_loss = self.residual_vector_quantize(z_e_down, num_streams)

        if freeze: z_q_down = z_e_down + z_q_down*0.

        z_q = self.proj_up(z_q_down) if self.do_proj else z_q_down
        z_q = post_process(z_q, self.in_freq, self.overlap, self.fix_dim, dims) # [B, H*W, C] / [B, C, H, W]

        return (z_q, indices), (commitment_loss, codebook_loss)

class ProductVectorQuantize(nn.Module):
    "Product VQ Layer"
    def __init__(self, 
                in_dim: int,
                in_freq: int, 
                overlap: int=4,
                num_vqs: int=3, 
                codebook_dim: int=8,
                codebook_size: int=1024, 
                l2norm: bool=True,
                kmeans_init: bool=False) -> None:
        super().__init__()

        self.overlap, self.codebook_dim, self.codebook_size, self.num_vqs = overlap, codebook_dim, codebook_size, num_vqs
        self.in_freq, self.in_dim = in_freq, in_dim

        self.fix_dim = in_freq*in_dim # dimension after reshaping
        self.vq_dims = split_dimension(self.fix_dim*overlap, num_vqs)
        self.vqs = nn.ModuleList([ 
                Codebook(in_dim, codebook_dim, codebook_size, l2norm) for in_dim in self.vq_dims
            ]) 
        
        self.kmeans_init = kmeans_init
        self.verbose_init = False # True for verbosing after initialization
        self.register_buffer('codebook_initialized', torch.zeros(1)) 
        # random initialization when set to None
        # initialized by kaiming normal by default (set to False)
        # initialized by kmeans after pre-training (set to True)
        self.register_forward_hook(codebook_init_forward_hook_pvq)

    def forward(self, z_e, freeze=False):
        """ Product VQ Forwrd Function.
        Args: 
            z_e: Input vector with shape (B, H*W, C) [swinT output] or (B, C, H, W) [conv output]
            freeze: boolean (True for handling pre-training stage, when codebook is not updated)
        """
        dims = len(z_e.shape)
        z_e = pre_process(z_e, self.in_freq, self.overlap, self.fix_dim, dims) # [B, W//overlap, overlap*H*C]
        z_q, z_e_downs, indices = [], [], []
        codebook_loss, commitment_loss = 0., 0.
        
        s_idx = 0
        for i, vq in enumerate(self.vqs):
            e_idx = s_idx + self.vq_dims[i]
            outputs = vq(z_e[..., s_idx:e_idx], freeze)
            z_q_i, z_e_down_i, code_i = outputs["z_q"], outputs["z_e_down"], outputs["code"]
            cm_loss, cb_loss = outputs["commitment_loss"], outputs["codebook_loss"]

            indices.append(code_i)
            z_q.append(z_q_i)
            z_e_downs.append(z_e_down_i)

            commitment_loss += cm_loss
            codebook_loss += cb_loss
            s_idx = e_idx

        z_q = post_process(torch.cat(z_q, dim=-1),
                self.in_freq, self.overlap, self.fix_dim, dims) # [B, H*W, C] / [B, C, H, W]      
        indices = torch.stack(indices, dim=1)                   # [B, group_size, T]
        z_e_downs = torch.stack(z_e_downs, dim=1)               # [B, group_size, T, codebook_dim] (used for kmeans)
        return (z_q, z_e_downs, indices), (commitment_loss/self.num_vqs, codebook_loss/self.num_vqs)

    def encode(self, z_e):
        """ Encode to codes
        Args:
            returns: indices of size (B, group_size, T)
        """

        z_e = pre_process(z_e, self.in_freq, self.overlap, self.fix_dim, len(z_e.shape))
        s_idx, codes = 0, []
        for i, vq in enumerate(self.vqs):
            e_idx = s_idx + self.vq_dims[i]
            code = vq.encode(z_e[..., s_idx:e_idx])
            codes.append(code)
            s_idx = e_idx

        codes = torch.stack(codes, dim=1)
        return codes
    
    def decode(self, codes, dims=3):
        """ Decode from codes
        Args:
            codes: indices tensor of size (B, Group_size, T)
            dims: 3 for swinT / 4 for conv
            returns: reconstructed vector
        """
        z_q = []
        for i, vq in enumerate(self.vqs):
            code = codes[:, i:i+1, :]
            z_q_i = vq.decode(code)
            z_q.append(z_q_i)

        z_q = post_process(torch.cat(z_q, dim=-1),
                self.in_freq, self.overlap, self.fix_dim, dims)
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
        z_e: Input vector with shape (B, H*W, C) [swinT output] or (B, C, H, W) [conv output]
        in_freq: H
        overlap: Number of overlapped frames to quantize together
        fix_dim: C*H
        dims: 3 for swinT / 4 for conv
        returns: Reshaped vector with shape (B, W//overlap, overlap*H*C)
    """
    if dims == 3:   # [swinT output]
        z_e = rearrange(z_e, "b (h w) c -> b w (c h)", h=in_freq)
    elif dims == 4: # [conv output]
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
        z_q: Quantized vector with shape (B, W//overlap, overlap*H*C)
        in_freq: H
        overlap: Number of overlapped frames to quantize together
        fix_dim: C*H
        dims: 3 for swinT / 4 for conv
        returns: Recovered vector with shape (B, H*W, C) [swinT output] or (B, C, H, W) (conv output)
    """
    # split overlapping frames
    if overlap > 1:
        z_q = z_q.view(z_q.size(0), -1, overlap, fix_dim) \
            .reshape(z_q.size(0), -1, fix_dim)
    if dims == 3:   # [swinT output]
        z_q = rearrange(z_q, "b w (c h) -> b (h w) c", h=in_freq)
    elif dims == 4: # [conv output]
        z_q = rearrange(z_q, "b w (c h) -> b c h w", h=in_freq)

    return z_q