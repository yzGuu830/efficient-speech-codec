import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, hidden_size, res_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(hidden_size,hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_size, res_size, 3, 1, 1),
            nn.GroupNorm(res_size,res_size),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(res_size, hidden_size, 1, 1, 0),
        )

    def forward(self, input):
        return input + self.conv(input)
        
    
    