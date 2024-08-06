import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Codebook(nn.Module):
    def __init__(self, 
                 embedding_dim: int=256, 
                 num_embeddings: int=1024, 
                 l2norm: bool=False,
                 ):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.kaiming_normal_(self.embedding.weight) 	
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.l2norm = l2norm

    def quantize_to_code(self, z_e):
        """ Quantize input vector to codebook indices.
        Args:
            z_e (Tensor): input vector with shape (bs, *, embedding_dim)
        Returns: 
            Tensor of indices with shape (bs, *)
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
        Args:
            code (Tensor): code with shape (bs, *)
        Returns:
            Tensor of quantized vector with shape (bs, *, embedding_dim)
        """
        codebook = self.embedding.weight
        z_q = F.embedding(code, codebook)

        return z_q

    def forward(self, z_e):
        """ Vector Quantization Forward Function.
        Args:
            z_e (Tensor): input vector with shape (bs, T, embedding_dim)
            z_q (Tensor): quantized vector with shape (bs, T, embedding_dim)
        """              

        code = self.quantize_to_code(z_e)
        z_q = self.dequantize_code(code)
        
        if self.training: # Straight-Through Estimator
            commitment_loss = F.mse_loss(z_q.detach(), z_e, reduction="none").mean([1,2])
            codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1,2])
            z_q = z_e + (z_q - z_e).detach()
        else:
            commitment_loss = F.mse_loss(z_q, z_e, reduction="none").mean([1,2])
            codebook_loss = commitment_loss
                                
        return z_q, code, codebook_loss, commitment_loss

    def encode(self, z_e):
        code = self.quantize_to_code(z_e)
        return code
    
    def decode(self, code):
        z_q = self.dequantize_code(code)
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
