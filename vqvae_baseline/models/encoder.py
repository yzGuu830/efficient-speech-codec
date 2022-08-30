import torch
import torch.nn as nn
import numpy as np
from models.residual import ResidualStack



class Encoder(nn.Module):
    '''
    Encoder Network: Z_e(x)
    This maps data sample x to latent space z
    The network follows a simple CNN downsampling structure
    '''

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim) -> None:
        super(Encoder,self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim//2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim//2,  h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self,x):
        return self.conv_stack(x)




if __name__ == "__main__":
    # Test

    x = np.random.random_sample((3,40,40,200))
    x = torch.tensor(x).float()

    encoder = Encoder(in_dim=40,h_dim=128,n_res_layers=3,res_h_dim=64)
    Z_e = encoder(x)

    print(f'Encoder output shape: {Z_e.shape}')
