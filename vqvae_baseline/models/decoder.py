import torch
import torch.nn as nn
import numpy as np
from models.residual import ResidualStack



class Decoder(nn.Module):
    '''
    Decoder Network: Z_d(x)
    This maps quantized latent feature z_q(x) to x'
    The network follows a simple CNN upsampling structure
    '''

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim) -> None:
        super(Decoder,self).__init__()
        kernel = 4
        stride = 2
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim,  h_dim//2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, 1, kernel_size=kernel, stride=stride, padding=1),
            
        )

    def forward(self,x):
        return self.inverse_conv_stack(x)




if __name__ == "__main__":
    # Test

    x = np.random.random_sample((3,40,40,200))
    x = torch.tensor(x).float()

    decoder = Decoder(in_dim=40,h_dim=128,n_res_layers=3,res_h_dim=64)
    Z_d = decoder(x)

    print(f'Encoder output shape: {Z_d.shape}')
