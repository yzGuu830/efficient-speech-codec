import torch.nn as nn

class Convolution2D(nn.Module):
    """2D Convolution Layer"""
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size=(5,2),
                 scale=True,
                 transpose=False):
        super().__init__()

        stride = (2,1) if scale else (1,1)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(2,1)) if not transpose \
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=(1,0))
        self.conv = conv
        self.transpose, self.scale = transpose, scale

    def forward(self, x):
        F, T = x.size(-2), x.size(-1)
        y = self.conv(x)
        
        if self.scale:
            y = y[..., :F*2, :T] if self.transpose else y[..., :F//2, :T]
        else:
            y = y[..., :F, :T]

        return y

class ResidualUnit(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.block = nn.Sequential(*[
            Convolution2D(dim, dim, kernel_size=(5,2), scale=False),
            nn.BatchNorm2d(dim),
            nn.PReLU(),
            Convolution2D(dim, dim, kernel_size=(5,2), scale=False),
            nn.BatchNorm2d(dim),
            nn.PReLU(),
        ])

    def forward(self, x):
        y = self.block(x)

        return x + y


class ConvolutionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, depth=1,
                       kernel_size=(5,2), transpose=False) -> None:
        super().__init__()
        
        blocks = [ResidualUnit(in_dim) for _ in range(depth)]
        blocks += [Convolution2D(in_dim, out_dim, kernel_size, scale=True, transpose=transpose),
                   nn.BatchNorm2d(out_dim),
                   nn.PReLU(),]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        
        y = self.blocks(x)
        return y