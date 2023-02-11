import torch
import torch.nn as nn
import torch.nn.functional as F
from models.quantizer import Group_Quantize

from models.utils import fold, unfold

from config import cfg
from models.utils import find_nearest


class Fuse_VQ(nn.Module):

    """This is an implementation of Fuse-VQ-Refine Module as figure2(b) in https://arxiv.org/abs/2207.03067"""

    def __init__(self, num_groups, input_channel, F_dim, down_rate, codebook_size, vq_commit):
        super().__init__()
        # assert C_down_rate <= 0.5
        self.num_groups  = num_groups
        self.input_channel = input_channel
        self.c_fix = find_nearest(int(self.input_channel * F_dim * down_rate), self.num_groups) # this is to make sure 4*c_fix is a multiple of num_groups
        
        # Group Quantizer
        self.quantizer = Group_Quantize(num_groups=self.num_groups, K=4*self.c_fix//self.num_groups, codebook_size=codebook_size, vq_commit=vq_commit)

        # Conv Blocks
        self.TwodConvin = nn.Sequential(*[nn.Conv2d(self.input_channel*2, self.input_channel, 3, 1, 1), 
                                          nn.Conv2d(self.input_channel, self.input_channel, 3, 1, 1)])
        self.down_channel_conv = nn.Conv1d(input_channel*F_dim, self.c_fix, 1, 1)

        self.TwodConvout = nn.Sequential(*[nn.Conv2d(self.input_channel*2, self.input_channel, 3, 1, 1), 
                                          nn.Conv2d(self.input_channel, self.input_channel, 3, 1, 1)])
        self.up_channel_conv = nn.Conv1d(self.c_fix, input_channel*F_dim, 1, 1)

    def forward(self, z_q, z_e, transmit=True):
        if z_q.size(2) < z_e.size(2):
            z_q = torch.nn.functional.pad(z_q,(0,0,1,0))
        elif z_q.size(2) > z_e.size(2):
            z_e = torch.nn.functional.pad(z_e,(0,0,1,0))
        '''
        input: 
        z_q: decoded feature (N, C, F, T) 
        z_e: encoded feature (N, C, F, T)
        returns:
        z_q_: refined decoded feature (N, C, F, T)
        '''

        if not transmit:
            q_merge = torch.zeros(size=z_q.size(), device=cfg['device'])
            vq_loss = torch.tensor(0.0, device=cfg['device'])

        else:
            # 1. concat
            z_merge = torch.cat((z_q,z_e), dim=1)            # (N, 2C, F, T)
            # 2. 2DConvolution Block (downsample 1C)
            z_merge = self.TwodConvin(z_merge)               # (N, C, F, T)
            # 3. 1DConvolution downsample
            z_merge = self.down_channel_conv(fold(z_merge))  # (N, C, F, T) -> (N, C*F, T) -> (N, C', T)

            # 4. Group Quantize
            q_merge, vq_loss = self.quantizer(z_merge)       # (N, C', T)^

            # 5. 1DConvolution upsample
            q_merge = unfold(self.up_channel_conv(q_merge), self.input_channel)  # (N, C', T)^ -> (N, C*F, T)^ -> (N, C, F, T)

        # 6. concat
        z_q_ = torch.cat((z_q,q_merge), dim=1)           # (N, 2C, F, T)
        # 7. 2DConvolution Block (downsample 1C)
        z_q_ = self.TwodConvout(z_q_)                    # (N, C, F, T)

        return z_q_, vq_loss







