import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from torch.nn.utils import weight_norm


class Codebook(nn.Module):
    def __init__(self, 
                 input_size: int=512,
                 embedding_size: int=256, 
                 num_embedding: int=1024, 
                 use_cosine_sim: bool=False,
                 ):
        super().__init__()

        if embedding_size is None:
            embedding_size = input_size

        self.embedding_size = embedding_size
        self.num_embedding = num_embedding
        self.embedding = nn.Embedding(num_embedding, embedding_size)

        self.use_cosine_sim = use_cosine_sim

        self.proj = (input_size != embedding_size)
        if self.proj:
            self.proj_down = (
                nn.Linear(input_size, embedding_size, bias=False)
            )
            self.proj_up = (
                nn.Linear(embedding_size, input_size, bias=False)
            )

    def quantize_to_code(self, z):
        """
            codebook: [n, embedding_size]
            z: [bs, T, embedding_size]
            returns: [bs, T]
        """

        codebook = self.embedding.weight                # [num_embedding, embedding_size]
        z_flat = rearrange(z, "b t d -> (b t) d")       # [bs*T, embedding_size]

        if self.use_cosine_sim:
            codebook = F.normalize(codebook, dim=-1)
            z_flat = F.normalize(z_flat, dim=-1)
        
        dist = ( 
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t() 
            )
        indices = dist.min(1).indices
        indices = rearrange(indices, "(b t) -> b t", b=z.size(0))

        return indices

    def dequantize_code(self, code):
        """
            code: [bs, T]
            returns: [bs, T, embedding_size]
        """

        codebook = self.embedding.weight
        z_q = F.embedding(code, codebook)

        return z_q

    def forward(self, z):
        """ Forward Training
            z: [bs, T, embedding_size]
            z_q: [bs, T, embedding_size]^
        """

        z_e = self.proj_down(z) if self.proj else z

        code = self.quantize_to_code(z_e)
        z_q = self.dequantize_code(code)

        commitment_loss = F.mse_loss(z_q.detach(), z_e, reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        if self.training:
            z_q = z_e + (z_q - z_e).detach()
        
        z_q = self.proj_up(z_q) if self.proj else z_q

        return z_q, commitment_loss, codebook_loss, code
    
    def encode(self, z):
        z_in = self.proj_down(z) if self.proj else z
        code = self.quantize_to_code(z_in)
        return code
    
    def decode(self, code):
        z_q = self.dequantize_code(code)
        z_q_out = self.proj_up(z_q) if self.proj else z_q
        return z_q_out


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = weight_norm(
                nn.Linear(input_dim, codebook_dim, bias=False)
            )
        self.out_proj = weight_norm(
                nn.Linear(codebook_dim, input_dim, bias=False)
            )
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x T x D]

        Returns
        -------
        Tensor[B x T x D]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x T x D]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x T x D)
        print("down z: ", z_e)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b t d -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        print("dist: ", dist)
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices



class CodebookEMA(Codebook):

    def __init__(self, 
                input_size: int = 512, 
                embedding_size: int = 256, 
                num_embedding: int = 1024, 
                use_cosine_sim: bool = False, 
                decay: float = 0.99, eps: float = 0.00001):
        super().__init__(input_size, embedding_size, num_embedding, use_cosine_sim)

        self.decay = decay
        self.eps = eps
        embedding = torch.randn(self.embedding_size, self.num_embedding, requires_grad=False)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(self.num_embedding))
        self.register_buffer('embedding_mean', embedding.clone())

    def quantize_to_code(self, z):
        """
        codebook: [n, embedding_size]
        z: [bs, T, embedding_size]
        """

        codebook = self.embedding.transpose(0,1)    # [num_embedding, embedding_size]
        z_flat = z.view(-1, self.embedding_size)    # [bs*T, embedding_size]

        if self.use_cosine_sim:
            codebook = F.normalize(codebook, dim=-1)
            z_flat = F.normalize(z_flat, dim=-1)
        
        dist = ( 
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t() 
            )

        indices = dist.min(1).indices
        indices = indices.view(z.shape[:-1])

        z_flat = z.view(-1, self.embedding_size)
        dist = ( z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ codebook.t()
                + codebook.pow(2).sum(1, keepdim=True).t() 
            )

        embed_idx = dist.min(1).indices
        code = embed_idx.view(*z.shape[:-1])

        return code, embed_idx, z_flat


    def dequantize_code(self, code):
        """code: [bs, T]"""

        codebook = self.embedding.transpose(0,1)
        z_q = F.embedding(code, codebook)

        return z_q


    def forward(self, z):


        # if self.training:
        #     with torch.no_grad():
        #         # Optimize Codebook via Exponential Moving Average
        #         embedding_onehot = F.one_hot(embed_idx, self.num_embedding).type(z.dtype)
        #         self.cluster_size.data.mul_(self.decay).add_(embedding_onehot.sum(0), alpha=1 - self.decay)
        #         embedding_sum = z_flat.transpose(0, 1) @ embedding_onehot
        #         self.embedding_mean.data.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)
        #         n = self.cluster_size.sum()
        #         cluster_size = (
        #                 (self.cluster_size + self.eps) / (n + self.num_embedding * self.eps) * n
        #         )
        #         embedding_normalized = self.embedding_mean / cluster_size.unsqueeze(0)
        #         self.embedding.data.copy_(embedding_normalized)

        return
