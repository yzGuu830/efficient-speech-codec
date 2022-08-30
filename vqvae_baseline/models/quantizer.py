import torch
import torch.nn as nn
import numpy as np
from utils import to_device

class Quantizer(nn.Module):
    '''
    bottleneck discretization


    Inputs:
    - n_e: number of vectors in codebook
    - e_dim: dimension of embedding vector
    - beta: committment loss tunable hyperparameter
    '''

    def __init__(self, n_e, e_dim, beta) -> None:
        super(Quantizer,self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.codebook = nn.Embedding(self.n_e, self.e_dim) # (n_e, e_dim)
        self.codebook.weight.data.uniform_(-1.0/self.n_e, 1.0/self.n_e)

    def forward(self, z):
        '''
        Z_e(x) -> Z_q(x)

        z.shape = (batch, channel, height, width) (B,C,H,W)

        '''

        # 1. reshape z and flatten
        z = z.permute(0,2,3,1).contiguous() # (B, H, W, C)
        z_flatten = z.view(-1,self.e_dim)  # (n, e_dim)

        # 2. calculate distance between z and Codebook e
        # distance = (z - e)^2 = z^2 + e^2 - 2*e*z

        dist =  torch.sum(z_flatten ** 2, dim=1, keepdim=True) + \
                torch.sum(self.codebook.weight**2, dim=1) - \
                2 * torch.matmul(z_flatten, self.codebook.weight.t())
        # (n, n_e)

        # 3. get closest vectors
        min_index = torch.argmin(dist, dim=1).unsqueeze(1)
        # to_device(min_index, 'cuda')

        # use one-hot matrix
        min_codes = torch.zeros(min_index.shape[0],self.n_e).to('cuda')
        # to_device(min_codes, 'cuda')

        min_codes.scatter_(1, min_index, 1)

        # 4. now we get quantized z_e -> z_q
        z_q = torch.matmul(min_codes, self.codebook.weight).view(z.shape)

        # Loss [alignment loss & committment loss]
        # (sg[z_e(x)] - e)^2 + _beta * (z_e(x) - sg[e])^2
        loss = torch.mean((z.detach() - z_q)**2) + self.beta * \
            torch.mean((z - z_q.detach()) ** 2)

        # preserve gradients ? 
        z_q = z + (z_q - z).detach()

        # perplexity ?
        e_mean = torch.mean(min_codes, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()


        return loss, z_q, min_codes, min_index, perplexity