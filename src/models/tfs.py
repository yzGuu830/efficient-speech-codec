import torch
import torch.nn as nn

from models.convolution import Convolution1D

class RNNFilter(nn.GRU):

    def __init__(self, 
                 input_size, 
                 hidden_size,
                 num_layers,
                 bias=True,
                 batch_first=True,
                 dropout=0.,
                 bidirectional=False,):
        super().__init__(input_size, 
                 hidden_size,
                 num_layers,
                 bias=bias,
                 batch_first=batch_first,
                 dropout=dropout,
                 bidirectional=bidirectional,)
        
        self.num_layers = num_layers
        self.D = 2 if bidirectional else 1
    
    def forward(self, x):
        """x: [bs, hidden, T]"""
        h_ = torch.randn(self.D * self.num_layers, x.size(0), x.size(1), device=x.device)
        super().flatten_parameters()

        x = x.permute(0,2,1)
        output, _ = super().forward(x, h_)
        output = output.permute(0,2,1).contiguous()
        return output


class TCM(nn.Module):
    """An implementation of TCM Module in TCNN: Temporal convolutional neural network for real-time speech enhancement in the time domain"""

    def __init__(self, in_channels, fix_channels, dilations=(1,2,4,8)):
        super().__init__()

        self.block = nn.Sequential(*[TCNNResBlock(in_channels, fix_channels, d) for d in dilations])

    def forward(self, x):
        return self.block(x)
    
class TCNNResBlock(nn.Module):
    '''
    One Dilated Residual Block of TCNN
    '''
    def __init__(self, in_channels, fix_channels, dilation, filter_size=5):
        super().__init__()

        self.block = nn.Sequential(*[
            Convolution1D(in_channels, fix_channels, kernel_size=1, bias=False, causal=True),
        
            nn.PReLU(num_parameters=1),
            nn.BatchNorm1d(fix_channels),

            DConv1d(fix_channels, fix_channels, kernel_size=filter_size, stride=1,    # Depthwise Causal Convolution 
                                   dilation=dilation, groups=fix_channels, bias=False),

            nn.PReLU(num_parameters=1),
            nn.BatchNorm1d(fix_channels),
            Convolution1D(fix_channels, in_channels, kernel_size=1, bias=False, causal=True)
        ])
        
    def forward(self, x):
        """x: [bs, hidden, T]"""
        
        return self.block(x) + x # residual


class DConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, dilation, groups, bias) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=(kernel_size - 1) * dilation, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x_ = self.conv(x)

        return x_[:, :, :-self.conv.padding[0]].contiguous()