import torch
import torch.nn as nn
from einops import rearrange

from modules.vq.codebook import Codebook
from modules.vq.initialize import codebook_init_forward_hook_pvq


class ProductVectorQuantize(nn.Module):
    "Product VQ Layer"
    def __init__(self, 
                in_dim: int,
                in_freq: int, 
                overlap: int=4,
                num_vqs: int=6, 
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
            z_e: Input vector, tensor with shape (B, H*W, C)
            freeze: boolean (True for handling pre-training stage, when codebook is not updated)
        """
        z_e = self.pre_process(z_e)
        z_q, z_e_downs, indices = [], [], []
        codebook_loss, commitment_loss = 0., 0.
        
        s_idx = 0
        for i, vq in enumerate(self.vqs):
            e_idx = s_idx + self.vq_dims[i]
            outputs, losses = vq(z_e[..., s_idx:e_idx], freeze)
            z_q_i, z_e_down_i, code_i = outputs
            cm_loss, cb_loss = losses

            indices.append(code_i)
            z_q.append(z_q_i)
            z_e_downs.append(z_e_down_i)

            commitment_loss += cm_loss
            codebook_loss += cb_loss
            s_idx = e_idx

        z_q = self.post_process(torch.cat(z_q, dim=-1)) # [B, H*W, C]       
        indices = torch.stack(indices, dim=1)           # [B, group_size, T]
        z_e_downs = torch.stack(z_e_downs, dim=1)       # [B, group_size, T, codebook_dim] (used for kmeans)
        return (z_q, z_e_downs, indices), (commitment_loss/self.num_vqs, codebook_loss/self.num_vqs)

    def pre_process(self, z_e):
        """ Pre-process input vector (reshaping and overlapping)
        Args: 
            z_e: Input vector with shape (B, H*W, C)
            returns: Reshaped vector with shape (B, W//overlap, overlap*H*C)
        """
        z_e = rearrange(z_e, "b (h w) c -> b w (c h)", h=self.in_freq)
        B, W = z_e.size(0), z_e.size(1)

        # overlap feature frames
        if self.overlap > 1:
            assert W % self.overlap == 0, "Time dimension must be multiple of overlap"
            z_e = z_e.view(B, W//self.overlap, self.overlap, self.fix_dim) \
                .reshape(B, W//self.overlap, self.overlap*self.fix_dim)    
        return z_e
    
    def post_process(self, z_q):
        """ Post-process quantized vector
        Args: 
            z_q: Quantized vector with shape (B, W//overlap, overlap*H*C) 
            returns: Recovered vector with shape (B, H*W, C)
        """
        # split overlapping frames
        if self.overlap > 1:
            z_q = z_q.view(z_q.size(0), -1, self.overlap, self.fix_dim) \
                .reshape(z_q.size(0), -1, self.fix_dim)
        
        z_q = rearrange(z_q, "b w (c h) -> b (h w) c", h=self.in_freq)
        return z_q

    def encode(self, z_e):
        """ Encode to codes
        Args:
            returns: indices of size (B, group_size, T)
        """

        z_e = self.pre_process(z_e)
        s_idx, codes = 0, []
        for i, vq in enumerate(self.vqs):
            e_idx = s_idx + self.vq_dims[i]
            code = vq.encode(z_e[..., s_idx:e_idx])
            codes.append(code)
            s_idx = e_idx

        codes = torch.stack(codes, dim=1)
        return codes
    
    def decode(self, codes):
        """ Decode from codes
        Args:
            codes: indices tensor of size (B, Group_size, T)
            returns: reconstructed vector
        """
        z_q = []
        for i, vq in enumerate(self.vqs):
            code = codes[:, i:i+1, :]
            z_q_i = vq.decode(code)
            z_q.append(z_q_i)

        z_q = self.post_process(torch.cat(z_q, dim=-1))
        return z_q

def split_dimension(total_dim, num):
    if total_dim % num == 0:
        dims = [total_dim//num for _ in range(num)]
    else:
        dims = [total_dim//num for _ in range(num-1)]
        dims += [total_dim - sum(dims)]
    return dims