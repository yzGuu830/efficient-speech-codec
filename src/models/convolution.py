import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class ConvEncoderLayer(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size=(5,2),
                 down = True,
                 norm = "batchnorm", 
                 act = "p_relu"):
        super().__init__()

        stride = (2, 1) if down else (1, 1)
        self.conv = Convolution2D(in_channels, 
                                  out_channels, 
                                  kernel_size,
                                  stride=stride, 
                                  causal=True)

        self.norm = nn.BatchNorm2d(out_channels) if norm == "batchnorm" else nn.Identity()

        self.act = nn.PReLU(num_parameters=1) if act == "p_relu" else nn.Identity()

    def forward(self,x):
        "x_shape: B, C=2, F, T"
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        "x_shape: B, C=2, F//2, T"
        return x

class ConvDecoderLayer(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,
                 kernel_size=(5,2),
                 up = True,
                 norm = "batchnorm", 
                 act = "p_relu"):
        super().__init__()

        stride = (2, 1) if up else (1, 1)
        self.conv = ConvolutionTranspose2D(in_channels, 
                                           out_channels, 
                                           kernel_size,
                                           stride=stride, 
                                           causal=True)

        self.norm = nn.BatchNorm2d(out_channels) if norm == "batchnorm" else nn.Identity()

        self.act = nn.PReLU(num_parameters=1) if act == "p_relu" else nn.Identity()

    def forward(self,x):
        "x_shape: B, C=2, F//2, T"
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        "x_shape: B, C=2, F, T"
        
        return x

class Convolution1D(nn.Module):
    """1D Convolution (dilated-causal convolution)"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 causal=True):
        super().__init__()
        # super().__init__(
        #     in_channels, out_channels, kernel_size,
        #     stride=stride, padding=0 if causal else padding,
        #     dilation=dilation, groups=groups, bias=bias)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0 if causal else padding,
            dilation=dilation, groups=groups, bias=bias
        )

        self.left_pad = dilation * (kernel_size - 1) if causal else 0

    def forward(self, input):
        x = F.pad(input, (self.left_pad, 0))

        # return super().forward(x)
        return self.conv(x)
    

class Convolution2D(nn.Module):
    """2D Convolution (dilated-causal convolution)"""

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=None, 
                 dilation=1, 
                 groups=1, 
                 bias=True,
                 causal=True):
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        if causal:
            padding = [int((kernel_size[i]-1) * dilation[i]) for i in range(len(kernel_size))]
    
        # super().__init__(
        #     in_channels, out_channels, kernel_size,
        #     stride=stride, padding=0 if causal else padding, 
        #     dilation=dilation, groups=groups, bias=bias)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0 if causal else padding, 
            dilation=dilation, groups=groups, bias=bias
        )
        
        self.left_pad = _pair(padding) if causal else _pair(0)

    def forward(self, inputs):
        x = F.pad(inputs, (self.left_pad[1], 0, self.left_pad[0], 0))

        # return super().forward(x)
        return self.conv(x)

class ConvolutionTranspose2D(nn.Module):
    """2D Transposed Convolution (dilated-causal convolution)"""

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=None,
                 dilation=1, 
                 groups=1, 
                 bias=True, 
                 causal=False):
        super().__init__()
        # super().__init__(
        #     in_channels, out_channels, kernel_size, stride=stride,
        #     padding=0 if causal else padding,
        #     dilation=dilation, groups=groups, bias=bias)
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=0 if causal else padding,
            dilation=dilation, groups=groups, bias=bias
        )
        self.causal = causal
        
        kh, kw = _pair(kernel_size)
        dh, dw = _pair(dilation)
        
        self.crop_h = (kh - 1) * dh if causal else 0
        self.crop_w = (kw - 1) * dw if causal else 0
    
    def forward(self, x):

        # x_ = super().forward(x)
        x_ = self.conv(x)
        
        if self.causal:
            x_ = x_[:, :, self.crop_h:, self.crop_w:].contiguous()

        h_pad, w_pad = x.shape[2] * self.conv.stride[0] - x_.shape[2], x.shape[3] * self.conv.stride[1] - x_.shape[3]
        x_ = F.pad(x_, (0, w_pad, 0, h_pad))
        
        return x_
    


if __name__ == "__main__":
    import torch
    x = torch.randn(1, 2, 192, 600)

    # 16,16,24,24,32,64

    encoder = nn.ModuleList([
        TempConv(2, 16, down=False),
        TempConv(16, 24, down=True),
        TempConv(24, 24, down=True),
        TempConv(24, 32, down=True),
        TempConv(32, 64, down=True),
    ])

    for i, mod in enumerate(encoder):
        x = mod(x)
        print(i, x.shape)

    decoder = nn.ModuleList([
        TempConvTranspose(64, 32, up=True),
        TempConvTranspose(32, 24, up=True),
        TempConvTranspose(24, 24, up=True),
        TempConvTranspose(24, 16, up=True),
        TempConvTranspose(16, 2, up=False)
    ])
    for i, mod in enumerate(decoder):
        x = mod(x)
        print(i, x.shape)
