import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convolution import Convolution1D

class GroupVQ(nn.Module):

    def __init__(self, 
                in_dim: int,
                H: int, 
                proj: int = 2,
                overlap: int = 4,
                num_vqs: int = 6, 
                codebook_size: int = 1024, 
                vq_commit: float = .25) -> None:
        super().__init__()
        
        self.fix_dim = get_multiple(in_dim*H//proj, num_vqs) if proj > 1 else in_dim*H

        # self.proj_down = Convolution1D(in_dim*H, self.fix_dim, kernel_size=1, bias=False, causal=True) if proj > 1 else None
        # self.proj_up = Convolution1D(self.fix_dim, in_dim*H, kernel_size=1, bias=False, causal=True) if proj > 1 else None
        # 1d convolution with kernelsize 1 equivalent to nn.linear
        self.proj_down = nn.Linear(in_dim*H, self.fix_dim, bias=False) if proj > 1 else None
        self.proj_up = nn.Linear(self.fix_dim, in_dim*H, bias=False) if proj > 1 else None

        self.overlap = overlap
        self.H, self.in_dim = H, in_dim

        self.vq_dims = check_remainder(self.overlap * self.fix_dim, num_vqs)
        self.vqs = nn.ModuleList([ VectorQuantization(
                                    self.vq_dims[i], codebook_size, vq_commit,
                                    ) 
                                    for i in range(num_vqs)])        
        
    def forward(self, z):

        z_ = self.pre_process(z)

        z_q, vq_loss = [], torch.tensor(0.0, device=z.device)

        s_idx = 0
        for i in range(len(self.vq_dims)):
            e_idx = s_idx + self.vq_dims[i]
            z_i = z_[:, :, s_idx:e_idx]

            z_q_i, vq_loss_i = self.vqs[i](z_i)

            z_q.append(z_q_i)
            vq_loss += vq_loss_i
            s_idx = e_idx
        
        z_q = torch.cat(z_q, dim=-1)
        z_q_ = self.post_process(z_q, dim=z.dim())

        return z_q_, vq_loss/len(self.vq_dims)

    def pre_process(self, z):
        """
        Args: z: should be either (B, C, H, W) as image or (B, H*W, C) as sequence
            vq requires input to be of shape (*, C)
            returns: (B, W//overlap, overlap*H*C//proj)
        """

        if z.dim() == 4:   # 2d output
            W = z.size(-1)
            assert z.size(2)==self.H and z.size(1)==self.in_dim, "z shape isn't correct"
            z = z.permute(0,2,3,1).contiguous()                              # B H W C
        elif z.dim() == 3: # 1d output
            W = z.size(1) // self.H
            assert z.size(2)==self.in_dim, "z shape isn't correct"
            z = z.view(z.size(0), self.H, W, self.in_dim).contiguous()  # B H W C
        else:
            raise ValueError("dim of z is not correct")

        # merge H and C
        # z = z.permute(0,2,1,3).reshape(z.size(0),W,self.H*self.in_dim)  # B W H*C
        z = z.permute(0,2,3,1).reshape(z.size(0),W,self.in_dim*self.H)  # B W C*H
        # projection
        if self.proj_down:
            z = self.proj_down(z)
        # overlap frames
        if self.overlap > 1:
            assert W % self.overlap == 0, "T dim must be multiple of overlap"
            z = z.view(z.size(0), W//self.overlap, self.overlap, self.fix_dim) \
                .reshape(z.size(0), W//self.overlap, self.overlap*self.fix_dim)
        return z
    
    def post_process(self, z_q, dim: int = 3):
        """
        Args: z_q: has size (B, W//overlap, overlap*H*C//proj) 
            returns: either (B, C, H, W) as image when dim = 4 or (B, H*W, C) as sequence when dim = 3
        """
        # overlap frames
        if self.overlap > 1:
            z_q = z_q.view(z_q.size(0), -1, self.overlap, self.fix_dim) \
                .reshape(z_q.size(0), -1, self.fix_dim)
        # projection
        if self.proj_up:
            z_q = self.proj_up(z_q) 
        # split H and C
        # z_q = z_q.reshape(z_q.size(0), -1, self.H, self.in_dim).permute(0,2,1,3) # B H W C
        z_q = z_q.reshape(z_q.size(0), -1, self.in_dim, self.H).permute(0,3,1,2) # B H W C

        if dim == 3:   # 1d output
            z_q = z_q.reshape(z_q.size(0), -1, z_q.size(-1))    # B H*W C
            assert z_q.size(2)==self.in_dim, "z_q shape isn't correct"

        elif dim == 4: # 2d output
            z_q = z_q.permute(0,3,1,2).contiguous()
            assert z_q.size(2)==self.H and z_q.size(1)==self.in_dim, "z_q shape isn't correct"

        return z_q

    def encode(self, z):

        z_ = self.pre_process(z)

        codes = []
        s_idx = 0
        for i in range(len(self.vq_dims)):
            e_idx = s_idx + self.vq_dims[i]
            z_i = z_[:, :, s_idx:e_idx]

            code, _, _ = self.vqs[i].quantize(z_i)
            codes.append(code)

            s_idx = e_idx

        return codes
    
    def decode(self, codes, dim: int=3):

        z_q = []
        for i, code in enumerate(codes):

            z_q_i = self.vqs[i].dequantize(code)
            z_q.append(z_q_i)

        z_q = torch.cat(z_q, dim=-1)

        z_q_ = self.post_process(z_q, dim)
        return z_q_


class VectorQuantization(nn.Module):
    """An implementation of VQ-VAE Quantizer trained by Exponential Moving Average Approach in https://arxiv.org/abs/1711.00937"""
    def __init__(self, embedding_size, num_embedding, vq_commit, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_embedding = num_embedding
        self.decay = decay
        self.eps = eps
        embedding = torch.randn(self.embedding_size, self.num_embedding, requires_grad=False)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(self.num_embedding))
        self.register_buffer('embedding_mean', embedding.clone())
        self.vq_commit = vq_commit

    def quantize(self, z):
        """z: [*, C=embedding_size]"""
        z_flat = z.view(-1, self.embedding_size)

        dist = ( z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.embedding
                + self.embedding.pow(2).sum(0, keepdim=True) )

        embed_idx = dist.min(1).indices
        code = embed_idx.view(*z.shape[:-1])

        return code, embed_idx, z_flat

    def dequantize(self, code):
        quantize = F.embedding(code, self.embedding.transpose(0, 1))
        return quantize

    def forward(self, z):
        """ Forward Training
        z: [bs, ..., embedding_size]
        """
        code, embed_idx, z_flat = self.quantize(z)
        z_q = self.dequantize(code)

        if self.training:
            with torch.no_grad():
                # Optimize Codebook via Exponential Moving Average
                embedding_onehot = F.one_hot(embed_idx, self.num_embedding).type(z.dtype)
                self.cluster_size.data.mul_(self.decay).add_(embedding_onehot.sum(0), alpha=1 - self.decay)
                embedding_sum = z_flat.transpose(0, 1) @ embedding_onehot
                self.embedding_mean.data.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cluster_size = (
                        (self.cluster_size + self.eps) / (n + self.num_embedding * self.eps) * n
                )
                embedding_normalized = self.embedding_mean / cluster_size.unsqueeze(0)
                self.embedding.data.copy_(embedding_normalized)
            z_q = z + (z_q - z).detach()
            
        vq_loss = self.vq_commit * F.mse_loss(z_q.detach(), z).mean()

        return z_q, vq_loss
    

def check_remainder(total_dim, num):
    if total_dim % num == 0:
        dims = [total_dim//num for _ in range(num)]
    else:
        dims = [total_dim//num for _ in range(num-1)]
        dims += [total_dim - sum(dims)]
    return dims


def get_multiple(n, d=6):
    while n % d != 0:
        n -= 1
    return n



if __name__ == "__main__":

    vq = GroupVQ(in_dim=64,
                H=12, 
                proj=2,
                overlap=4,
                num_vqs=6, 
                codebook_size=1024, 
                vq_commit= .25)
    vq.eval()
    print("Test Conv: ")
    with torch.inference_mode():
        z = torch.randn(1, 64, 12, 600)
        z_q, vq_loss = vq(z)
        codes = vq.encode(z)
        z_q_test = vq.decode(codes, dim=4)
        print(z_q.shape, z_q_test.shape, vq_loss)
        print(z_q-z_q_test)


    print("Test Swin")
    with torch.inference_mode():
        z = torch.randn(1, 12*600, 64)
        z_q, vq_loss = vq(z)
        codes = vq.encode(z)
        z_q_test = vq.decode(codes, dim=3)
        print(z_q.shape, z_q_test.shape, vq_loss)
        print(z_q-z_q_test)