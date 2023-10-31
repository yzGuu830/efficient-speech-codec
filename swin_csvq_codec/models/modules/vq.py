import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg

class GroupQuantization(nn.Module):
    """A Group Quantization Approach as proposed in https://arxiv.org/abs/1910.05453"""

    def __init__(self, 
                in_dim, 
                fix_dim, 
                dim_map=False,
                num_overlaps=4, 
                num_groups=6, 
                codebook_size=1024, 
                vq_commit=0.25):
        super().__init__()

        self.num_groups = num_groups
        self.dim_map = dim_map
        self.num_overlaps = num_overlaps

        if dim_map:
            self.fix_dim = fix_dim
            self.down_hidden_map = nn.Linear(in_dim, fix_dim)
            self.up_hidden_map = nn.Linear(fix_dim, in_dim)
        else:
            self.fix_dim = in_dim

        self.K = self.num_overlaps * self.fix_dim // self.num_groups

        self.Gruop_Quantizers = nn.ModuleList([
                                                VectorQuantization(
                                                    self.K, 
                                                    codebook_size, 
                                                    vq_commit) 
                                                    for _ in range(self.num_groups)
                                                ])
        self.training = True

    def pre_process(self, x, H, W):
        """x: [bs, H*W, C]"""

        x_2d = x.reshape(x.size(0), H, W, -1)      # [bs, H=F, W=T, C]

        x_1d = x_2d.permute(0,2,1,3).reshape(x.size(0), W, -1) # [bs, W=T, F*C]

        # x_1d = x

        if self.dim_map: 
            x_1d = self.down_hidden_map(x_1d)

        if self.num_overlaps > 1:
            x_1d = x_1d.view(x.size(0), W//self.num_overlaps, self.num_overlaps, self.fix_dim) \
                .reshape(x.size(0), W//self.num_overlaps, self.num_overlaps*self.fix_dim)

            # x_1d = x_1d.reshape(x.size(0), x.size(1)//self.num_overlaps, self.fix_dim*self.num_overlaps)

        assert x_1d.size(-1) == self.K * self.num_groups, "Dimension sum over Groups not Match with VQ"

        return x_1d


    def post_process(self, x_q, H, W):
        """x_q: [bs, W, H*C]"""

        if self.num_overlaps > 1:
            x_q = x_q.view(x_q.size(0), W//self.num_overlaps, self.num_overlaps, self.fix_dim) \
                .reshape(x_q.size(0), W, self.fix_dim)
            # x_q = x_q.reshape(x_q.size(0), x_q.size(1)*self.num_overlaps, self.fix_dim)

        if self.dim_map: 
            x_q = self.up_hidden_map(x_q) # [bs, W=T, F*C]

        x_q_2d = x_q.reshape(x_q.size(0), W, H, -1) # [bs, H=F, W=T, C]

        x_q_1d = x_q_2d.permute(0,2,1,3).reshape(x_q.size(0), H*W, -1) # [bs, H*W, C]

        return x_q_1d
        
    def quantize(self, z, H, W, verbose=False):
        '''
        z: feature [bs, H*W, d_model]
        returns: codes (a list of code)
        '''
        z_ = self.pre_process(z, H, W) # [bs, W, H*d_model]

        if verbose: 
            L, d = z_.size(1), z_.size(2)
        else:
            L, d = None, None

        codes = []
        for i in range(self.num_groups):
            z_i = z_[:, :, self.K*i:self.K*(i+1)]
            code, _, _ = self.Gruop_Quantizers[i].quantize(z_i)
            codes.append(code)
        return codes, (L, d)
        
    def dequantize(self, codes, H, W):
        q_merge = []
        for i in range(self.num_groups):
            quantize = self.Gruop_Quantizers[i].dequantize(codes[i])
            q_merge.append(quantize)

        q_merge = torch.cat(q_merge, dim=-1)

        return self.post_process(q_merge, H, W)

    def forward(self, z, H, W):
        ''' Group Quantization Training Forward 
        z: feature [bs, H*W, d_model]
        returns: quantized feature [bs, num_patch, d_model]^ & vq_loss over groups
        '''
        z_ = self.pre_process(z, H, W) # [bs, W, H*d_model]

        # Quantize feature by parts
        self.set_train()
        q_merge, vq_loss = [], torch.zeros(z.size(0), device=z.device) if cfg.num_workers > 1 else torch.tensor(0.0, device=z.device)
        for i in range(self.num_groups):
            z_i = z_[:, :, self.K*i:self.K*(i+1)]
            q_i, vq_loss_g, _ = self.Gruop_Quantizers[i](z_i)
            q_merge.append(q_i)
            vq_loss += vq_loss_g

        q_merge = torch.cat(q_merge, dim=-1)
        
        return self.post_process(q_merge, H, W), vq_loss/self.num_groups

    def set_train(self):
        for vqvae in self.Gruop_Quantizers:
            vqvae.training = True 
        return

class VectorQuantization(nn.Module):
    """An implementation of VQ-VAE Quantizer trained by Exponential Moving Average Approach in https://arxiv.org/abs/1711.00937"""
    def __init__(self, embedding_size, num_embedding, vq_commit, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_embedding = num_embedding
        self.decay = decay
        self.eps = eps
        embedding = torch.randn(self.embedding_size, self.num_embedding)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(self.num_embedding))
        self.register_buffer('embedding_mean', embedding.clone())
        self.vq_commit = vq_commit
        
    def quantize(self, input):
        """input: [bs, ..., embedding_size]"""
        flatten = input.reshape(-1, self.embedding_size)

        dist = ( flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embedding
                + self.embedding.pow(2).sum(0, keepdim=True) )

        _, embedding_ind = dist.min(1)

        code = embedding_ind.view(*input.shape[:-1])

        return code, embedding_ind, flatten

    def dequantize(self, code):

        quantize = self.embedding_code(code)

        return quantize

    def forward(self, input):
        """ Forward Training
        input: [bs, ..., embedding_size]
        """
        code, embedding_ind, flatten = self.quantize(input)
        quantize = self.dequantize(code)

        if self.training:
            # Optimize Codebook via Exponential Moving Average
            embedding_onehot = F.one_hot(embedding_ind, self.num_embedding).type(input.dtype)
            self.cluster_size.data.mul_(self.decay).add_(embedding_onehot.sum(0), alpha=1 - self.decay)
            embedding_sum = flatten.transpose(0, 1) @ embedding_onehot
            self.embedding_mean.data.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.num_embedding * self.eps) * n
            )
            embedding_normalized = self.embedding_mean / cluster_size.unsqueeze(0)
            self.embedding.data.copy_(embedding_normalized)

            quantize = input + (quantize - input).detach()
            
        diff = self.vq_commit * F.mse_loss(quantize.detach(), input, reduction='none').mean(dim=[1,2]) if cfg.num_workers > 1 \
            else self.vq_commit * F.mse_loss(quantize.detach(), input)

        return quantize, diff, code

    def embedding_code(self, embedding_ind):
        return F.embedding(embedding_ind, self.embedding.transpose(0, 1))
        
    def __repr__(self) -> str:
        return f"embedding_dim:{self.embedding_size}\nnum_embedding:{self.num_embedding}\nbeta:{self.vq_commit}"