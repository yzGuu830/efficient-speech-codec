import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import warnings

class Codebook(nn.Module):
    def __init__(self, 
                 input_dim: int=512,
                 embedding_dim: int=256, 
                 num_embeddings: int=1024, 
                 l2norm: bool=False,
                 ):
        super().__init__()

        if embedding_dim is None: # else perform projection
            embedding_dim = input_dim

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.l2norm = l2norm

        self.do_proj = (input_dim != embedding_dim)
        if self.do_proj:
            self.proj_down = nn.Linear(input_dim, embedding_dim, bias=False)
            self.proj_up = nn.Linear(embedding_dim, input_dim, bias=False)

    def quantize_to_code(self, z_e):
        """ Quantize input vector to codebook indices.
        Args:
            codebook: [N, embedding_dim]
            z_e: [bs, *, embedding_dim]
            returns: [bs, *]
        """

        codebook = self.embedding.weight                  # [num_embeddings, embedding_dim]
        z_flat = rearrange(z_e, "b t d -> (b t) d")       # [*, embedding_dim]

        if self.l2norm:
            codebook = F.normalize(codebook, dim=-1)
            z_flat = F.normalize(z_flat, dim=-1)
        
        dist = ( 
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t() 
            )
        indices = dist.min(1).indices
        indices = rearrange(indices, "(b t) -> b t", b=z_e.size(0))

        return indices

    def dequantize_code(self, code):
        """ De-quantize code indices to vectors
            code: [bs, *]
            returns: [bs, *, embedding_dim]
        """
        codebook = self.embedding.weight
        z_q = F.embedding(code, codebook)

        return z_q

    def forward(self, z_e, freeze=False):
        """ Forward Function.
            z_e: input vector [bs, T, embedding_dim]
            z_q: quantized vector [bs, T, embedding_dim]
            freeze: boolean (True for handling pre-training stage, when codebook is not updated)
        """
        if not self.training and freeze:     
            warnings.warn("Codebook Argument \{freeze\} must be set False during Inference") 
            freeze = False                

        z_e_down = self.proj_down(z_e) if self.do_proj else z_e
        code = self.quantize_to_code(z_e_down)
        z_q_down = self.dequantize_code(code)
        
        if self.training and freeze:            # Freeze Update to Codebook         
            z_q_down = z_e_down + z_q_down * 0.                 
            commitment_loss, codebook_loss = 0., 0.
        elif self.training and (not freeze):    # Straight-Through Estimator
            commitment_loss = F.mse_loss(z_q_down.detach(), z_e_down, reduction="none").mean([1,2])
            codebook_loss = F.mse_loss(z_q_down, z_e_down.detach(), reduction="none").mean([1,2])
            z_q_down = z_e_down + (z_q_down - z_e_down).detach() 
        elif not self.training:                 # Compute Loss at Test
            commitment_loss = F.mse_loss(z_q_down, z_e_down, reduction="none").mean([1,2])
            codebook_loss = commitment_loss
                        
        z_q = self.proj_up(z_q_down) if self.do_proj else z_q_down
        return (z_q, z_e_down, code), (commitment_loss, codebook_loss)
    

    def encode(self, z_e):
        z_e_down = self.proj_down(z_e) if self.do_proj else z_e
        code = self.quantize_to_code(z_e_down)
        return code
    
    def decode(self, code):
        z_q_down = self.dequantize_code(code)
        z_q = self.proj_up(z_q_down) if self.do_proj else z_q_down
        return z_q

def count_posterior(code, codebook_size):
    """ Compute the posterior codebook distribution P(q|e) on a total batch of encoded features
        Args:
            code: quantized discrete code of size [B, T]
            codebook_size: total number of entries
            returns: posterior distribution with size [B, codebook_size]
    """
    one_hot = F.one_hot(code, num_classes=codebook_size) # B T codebook_size
    counts = one_hot.sum(dim=1) # B codebook_size
    posterior = counts / code.size(1)

    return posterior
