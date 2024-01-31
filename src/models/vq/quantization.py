import torch
import torch.nn as nn
from einops import rearrange
from models.vq.codebook import Codebook, CodebookEMA

class ResidualVQ(nn.Module):
    def __init__(self, 
                in_dim: int,
                H: int, 
                overlap: int = 4,
                num_vqs: int = 6,
                codebook_dim: int = 8,
                codebook_size: int = 1024, 
                use_ema: bool = False,
                use_cosine_sim: bool = True,
                 ) -> None:
        super().__init__()

        codebook = Codebook if not use_ema else CodebookEMA
        if use_ema: use_cosine_sim = False
        self.overlap, self.codebook_dim, self.num_vqs = overlap, codebook_dim, num_vqs
        self.H, self.in_dim = H, in_dim
        self.fix_dim = H*in_dim
        self.vq_dims = [in_dim*H*overlap for _ in range(num_vqs)]
        self.vqs = nn.ModuleList([ 
                                    codebook(
                                        input_size=in_dim,
                                        embedding_size=codebook_dim, 
                                        num_embedding=codebook_size, 
                                        use_cosine_sim=use_cosine_sim,
                                        ) 
                                    for in_dim in self.vq_dims
                                ]) 
        
    def forward(self, z):

        dim = z.dim()

        z = self.pre_process(z)
        z_q, codebook_loss, commitment_loss = 0, \
            torch.zeros(z.size(0), device=z.device), \
                torch.zeros(z.size(0), device=z.device)
        
        residual = z
        for i, quantizer in enumerate(self.vqs):

            z_q_i, cm_loss, cb_loss, _ = quantizer(residual)

            z_q = z_q + z_q_i
            residual = residual - z_q_i

            commitment_loss += cm_loss
            codebook_loss += cb_loss

        z_q_ = self.post_process(z_q, dim=dim)
        return z_q_, commitment_loss, codebook_loss

    def pre_process(self, z):
        """
        Args: 
            z:  should be either (B, C, H, W) as image or (B, H*W, C) as sequence
                vq requires input to be of shape (*, C)
            returns: (B, W//overlap, overlap*H*C)
        """

        if z.dim() == 4:   # 2d output
            assert z.size(2)==self.H and z.size(1)==self.in_dim, "z shape isn't correct"
            z = rearrange(z, "b c h w -> b w (c h)")
        elif z.dim() == 3: # 1d output
            assert z.size(2)==self.in_dim, "z shape isn't correct"
            z = rearrange(z, "b (h w) c -> b w (c h)", h=self.H)
        else:
            raise ValueError("dim of z is not correct")

        W = z.size(1)

        # if self.proj_ratio < 1.0:
        #     z = self.in_proj(z)

        # overlap frames
        if self.overlap > 1:
            assert W % self.overlap == 0, "T dim must be multiple of overlap"
            z = z.view(z.size(0), W//self.overlap, self.overlap, self.fix_dim) \
                .reshape(z.size(0), W//self.overlap, self.overlap*self.fix_dim)
            
        return z
    
    def post_process(self, z_q, dim: int = 3):
        """
        Args: 
            z_q: has size (B, W//overlap, overlap*H*C) 
            returns: either (B, C, H, W) as image when dim = 4 or (B, H*W, C) as sequence when dim = 3
        """
        # overlap frames
        if self.overlap > 1:
            z_q = z_q.view(z_q.size(0), -1, self.overlap, self.fix_dim) \
                .reshape(z_q.size(0), -1, self.fix_dim)
        
        # if self.proj_ratio < 1.0:
        #     z_q = self.out_proj(z_q)
        
        if dim == 3:   # 1d output
            z_q = rearrange(z_q, "b w (c h) -> b (h w) c", h=self.H)
            assert z_q.size(2)==self.in_dim, "z_q shape isn't correct"

        elif dim == 4: # 2d output
            z_q = rearrange(z_q, "b w (c h) -> b c h w", h=self.H)
            assert z_q.size(2)==self.H and z_q.size(1)==self.in_dim, "z_q shape isn't correct"

        return z_q

class GroupVQ(nn.Module):

    def __init__(self, 
                in_dim: int,
                H: int, 
                overlap: int = 4,
                num_vqs: int = 6, 
                proj_ratio: float = 1.0,
                codebook_dim: int = 8,
                codebook_size: int = 1024, 
                use_ema: bool = False,
                use_cosine_sim: bool = True,) -> None:
        super().__init__()

        codebook = Codebook if not use_ema else CodebookEMA
        if use_ema: use_cosine_sim = False

        self.overlap, self.codebook_dim, self.num_vqs = overlap, codebook_dim, num_vqs
        self.H, self.in_dim = H, in_dim
        self.fix_dim = int(H*in_dim*proj_ratio)
        self.vq_dims = check_remainder(self.fix_dim*overlap, num_vqs)

        if proj_ratio < 1.0:
            self.in_proj = nn.Linear(H*in_dim, self.fix_dim, bias=False)
            self.out_proj = nn.Linear(self.fix_dim, H*in_dim, bias=False)
        self.proj_ratio = proj_ratio

        self.vqs = nn.ModuleList([ 
                                    codebook(
                                        input_size=in_dim,
                                        embedding_size=codebook_dim, 
                                        num_embedding=codebook_size, 
                                        use_cosine_sim=use_cosine_sim,
                                        ) 
                                    for in_dim in self.vq_dims
                                ]) 
                
    def forward(self, z):
        
        dim = z.dim()

        z = self.pre_process(z)
        z_q, codebook_loss, commitment_loss = [], \
            torch.zeros(z.size(0), device=z.device), \
                torch.zeros(z.size(0), device=z.device)

        s_idx = 0
        for i in range(len(self.vq_dims)):
            e_idx = s_idx + self.vq_dims[i]
            z_i = z[:, :, s_idx:e_idx]

            z_q_i, cm_loss, cb_loss, _ = self.vqs[i](z_i)

            z_q.append(z_q_i)
            commitment_loss += cm_loss
            codebook_loss += cb_loss

            s_idx = e_idx
        
        z_q = torch.cat(z_q, dim=-1)
        z_q_ = self.post_process(z_q, dim=dim)

        return z_q_, commitment_loss/len(self.vq_dims), codebook_loss/len(self.vq_dims)

    def pre_process(self, z):
        """
        Args: 
            z:  should be either (B, C, H, W) as image or (B, H*W, C) as sequence
                vq requires input to be of shape (*, C)
            returns: (B, W//overlap, overlap*H*C)
        """

        if z.dim() == 4:   # 2d output
            assert z.size(2)==self.H and z.size(1)==self.in_dim, "z shape isn't correct"
            z = rearrange(z, "b c h w -> b w (c h)")
        elif z.dim() == 3: # 1d output
            assert z.size(2)==self.in_dim, "z shape isn't correct"
            z = rearrange(z, "b (h w) c -> b w (c h)", h=self.H)
        else:
            raise ValueError("dim of z is not correct")

        W = z.size(1)

        if self.proj_ratio < 1.0:
            z = self.in_proj(z)

        # overlap frames
        if self.overlap > 1:
            assert W % self.overlap == 0, "T dim must be multiple of overlap"
            z = z.view(z.size(0), W//self.overlap, self.overlap, self.fix_dim) \
                .reshape(z.size(0), W//self.overlap, self.overlap*self.fix_dim)
            
        return z
    
    def post_process(self, z_q, dim: int = 3):
        """
        Args: 
            z_q: has size (B, W//overlap, overlap*H*C) 
            returns: either (B, C, H, W) as image when dim = 4 or (B, H*W, C) as sequence when dim = 3
        """
        # overlap frames
        if self.overlap > 1:
            z_q = z_q.view(z_q.size(0), -1, self.overlap, self.fix_dim) \
                .reshape(z_q.size(0), -1, self.fix_dim)
        
        if self.proj_ratio < 1.0:
            z_q = self.out_proj(z_q)
        
        if dim == 3:   # 1d output
            z_q = rearrange(z_q, "b w (c h) -> b (h w) c", h=self.H)
            assert z_q.size(2)==self.in_dim, "z_q shape isn't correct"

        elif dim == 4: # 2d output
            z_q = rearrange(z_q, "b w (c h) -> b c h w", h=self.H)
            assert z_q.size(2)==self.H and z_q.size(1)==self.in_dim, "z_q shape isn't correct"

        return z_q

    def encode(self, z):
        """
        Args:
            z: input to quantize
            returns: codes of size (B, Group_size, T)
        """

        z = self.pre_process(z)

        codes = [] # bs*group_size*T
        s_idx = 0
        for i in range(len(self.vq_dims)):
            e_idx = s_idx + self.vq_dims[i]
            z_i = z[:, :, s_idx:e_idx]

            code = self.vqs[i].encode(z_i) # bs*T
            codes.append(code)

            s_idx = e_idx

        codes = torch.stack(codes, dim=1)
        return codes # bs*group_size*T
    
    def decode(self, codes, dim: int=3):
        """
        Args:
            codes: discrete tensor of size (B, Group_size, T)
            returns: quantized representation
        """
        z_q = []
        for i in range(codes.size(1)):
            code = codes[:, i:i+1, :]
            z_q_i = self.vqs[i].decode(code)
            z_q.append(z_q_i)

        z_q = torch.cat(z_q, dim=-1) # B T D

        z_q_ = self.post_process(z_q, dim)
        return z_q_



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