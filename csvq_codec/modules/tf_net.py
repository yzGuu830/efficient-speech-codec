import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
import numpy as np


class CausalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(2,1), dilation=1) -> None:
        super().__init__()

        # causal along temporal dimension
        pad = (padding[0], (kernel_size[1] - 1) * dilation)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation)

    def forward(self, x):
        x_ = self.conv(x)

        return x_[:, :, :, :-self.conv.padding[1]].contiguous()


class CausalConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, padding=(2,1), dilation=1) -> None:
        super().__init__()

        # causal along temporal dimension
        pad = (padding[0], (kernel_size[1] - 1) * dilation)

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, output_padding=output_padding, dilation=dilation)

    def forward(self, x):
        x_ = self.conv(x)

        return F.pad(x_, pad=(0,1))

class F_downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2,1), causal=True):
        super().__init__()
        if causal:
            self.down = CausalConv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=(5,2), stride=stride, padding=(2,1))
        else:
            self.down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=(5,2), stride=stride, padding=(2,1))

        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(num_parameters=1)

    def forward(self,x):
        "x_shape: B, C=2, F, T"
        x = self.down(x)
        x = self.norm(x)
        x = self.act(x)
        "x_shape: B, C=2, F//2, T"

        return x

class F_upsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2,1), output_pad=(0,0), causal=True):
        super().__init__()

        if causal:
            self.up = CausalConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=(5,2), stride=stride, padding=(2,1), output_padding=output_pad)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=(5,2), stride=stride, padding=(2,1), output_padding=output_pad)

        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(num_parameters=1)

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

            CausalConv1d(fix_channels, fix_channels, kernel_size=filter_size, stride=1,    # Depthwise Causal Convolution 
                                   dilation=dilation, groups=fix_channels, bias=False),

            nn.PReLU(num_parameters=1),
            nn.BatchNorm1d(fix_channels),
            nn.Conv1d(fix_channels, in_channels, 1, 1)
        ])
        
    def forward(self, x):

        return self.block(x) + x # residual


class CausalConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, dilation, groups, bias) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=(kernel_size - 1) * dilation, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x_ = self.conv(x)

        return x_[:, :, :-self.conv.padding[0]].contiguous()