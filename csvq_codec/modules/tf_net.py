import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
import numpy as np


class CausalConv2d(nn.Conv2d):
    def __init__(self, *args, mask_type, data_channels, **kwargs):
        super(CausalConv2d, self).__init__(*args, **kwargs)

        assert mask_type in ['A', 'B'], 'Invalid mask type.'

        out_channels, in_channels, height, width = self.weight.size()
        yc, xc = height // 2, width // 2

        mask = np.zeros(self.weight.size(), dtype=np.float32)
        mask[:, :, :yc, :] = 1
        mask[:, :, yc, :xc + 1] = 1

        def cmask(out_c, in_c):
            a = (np.arange(out_channels) % data_channels == out_c)[:, None]
            b = (np.arange(in_channels) % data_channels == in_c)[None, :]
            return a * b

        for o in range(data_channels):
            for i in range(o + 1, data_channels):
                mask[cmask(o, i), yc, xc] = 0

        if mask_type == 'A':
            for c in range(data_channels):
                mask[cmask(c, c), yc, xc] = 0

        mask = torch.from_numpy(mask).float()

        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        x = super(CausalConv2d, self).forward(x)
        return x


class CausalConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, mask_type, data_channels, **kwargs):
        super(CausalConvTranspose2d, self).__init__(*args, **kwargs)

        assert mask_type in ['A', 'B'], 'Invalid mask type.'

        out_channels, in_channels, height, width = self.weight.size()
        yc, xc = height // 2, width // 2

        mask = np.zeros(self.weight.size(), dtype=np.float32)
        mask[:, :, :yc, :] = 1
        mask[:, :, yc, :xc + 1] = 1

        def cmask(out_c, in_c):
            a = (np.arange(out_channels) % data_channels == out_c)[:, None]
            b = (np.arange(in_channels) % data_channels == in_c)[None, :]
            return a * b

        for o in range(data_channels):
            for i in range(o + 1, data_channels):
                mask[cmask(o, i), yc, xc] = 0

        if mask_type == 'A':
            for c in range(data_channels):
                mask[cmask(c, c), yc, xc] = 0

        mask = torch.from_numpy(mask).float()

        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        x = super(CausalConvTranspose2d, self).forward(x)
        return x

class F_downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2,1)):
        super().__init__()
        
        self.down = CausalConv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=(5,3), stride=stride, padding=(2,1), 
                                mask_type='B', data_channels=in_channels)

        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        "x_shape: B, C=2, F, T"
        x = self.down(x)
        x = self.norm(x)
        x = self.act(x)
        "x_shape: B, C=2, F//2, T"

        return x

class F_upsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2,1), output_pad=(0,0)):
        super().__init__()

        self.up = CausalConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=(5,3), stride=stride, padding=(2,1), output_padding=output_pad,
                                mask_type='B', data_channels=in_channels)

        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        "x_shape: B, C=2, F//2, T"
        x = self.up(x)
        x = self.norm(x)
        x = self.act(x)
        "x_shape: B, C=2, F, T"
        
        return x


class TCM(nn.Module):
    """An implementation of TCM Module in TCNN: Temporal convolutional neural network for real-time speech enhancement in the time domain"""

    def __init__(self, in_channels, fix_channels, dilations=(1,2,4,8)):
        super().__init__()

        self.block = nn.Sequential(*[TCNN_ResBlock(in_channels, fix_channels, d) for d in dilations])

    def forward(self, x):

        return self.block(x)
    

class TCNN_ResBlock(nn.Module):
    '''
    One Dilated Residual Block of TCNN
    '''
    def __init__(self, in_channels, fix_channels, dilation, filter_size=5):
        super().__init__()

        self.block = nn.Sequential(*[
            nn.Conv1d(in_channels, fix_channels, 1, 1),
            nn.PReLU(num_parameters=1),
            nn.BatchNorm1d(fix_channels),

            nn.Conv1d(fix_channels, fix_channels, kernel_size=filter_size, stride=1, padding=(filter_size-1)*dilation//2,    # Depthwise Convolution 
                                   dilation=dilation, groups=fix_channels, bias=False),

            nn.PReLU(num_parameters=1),
            nn.BatchNorm1d(fix_channels),
            nn.Conv1d(fix_channels, in_channels, 1, 1)
        ])
        
    def forward(self, x):

        return self.block(x) + x # residual

