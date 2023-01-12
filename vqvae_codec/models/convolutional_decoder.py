from modules.residual_stack import ResidualStack
from error_handling.console_logger import ConsoleLogger

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers,
                    hierarchy_layer, num_residual_hiddens, verbose=False):
        super().__init__()
        self.hierarchy_layer = hierarchy_layer

        if self.hierarchy_layer == 1:
            block = [ # HVQVAE Layer1
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=num_hiddens,
                                    kernel_size=3,stride=1,padding=1),
                ResidualStack(hidden_size=num_hiddens, num_residual_layers=num_residual_layers, res_size=num_residual_hiddens),
                nn.ConvTranspose2d(in_channels=num_hiddens,out_channels=num_hiddens,
                                    kernel_size=4,stride=2,padding=1),
                nn.GroupNorm(num_hiddens,num_hiddens),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=num_hiddens,out_channels=num_hiddens//2,
                                    kernel_size=3,stride=1,padding=1),

                nn.GroupNorm(num_hiddens//2,num_hiddens//2),
                nn.LeakyReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=num_hiddens//2, out_channels=out_channels,
                                    kernel_size=4,stride=2,padding=1),
            ]
        elif self.hierarchy_layer == 2:
            block = [ # HVQVAE Layer2
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=num_hiddens,
                                    kernel_size=3,stride=1,padding=1),
                Residual(hidden_size=num_hiddens,res_size=num_residual_hiddens),
                Residual(hidden_size=num_hiddens,res_size=num_residual_hiddens),
                nn.ConvTranspose2d(in_channels=num_hiddens,out_channels=num_hiddens,
                                    kernel_size=4,stride=2,padding=1),
                nn.GroupNorm(num_hiddens,num_hiddens),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=num_hiddens,out_channels=num_hiddens//2,
                                    kernel_size=3,stride=1,padding=1),
                nn.GroupNorm(num_hiddens//2,num_hiddens//2),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(in_channels=num_hiddens//2, out_channels=num_hiddens,
                                    kernel_size=3,stride=1,padding=1),
                Residual(hidden_size=num_hiddens,res_size=num_residual_hiddens),
                Residual(hidden_size=num_hiddens,res_size=num_residual_hiddens),
                nn.ConvTranspose2d(in_channels=num_hiddens,out_channels=num_hiddens,
                                    kernel_size=4,stride=2,padding=1),
                nn.GroupNorm(num_hiddens,num_hiddens),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=num_hiddens,out_channels=num_hiddens//2,
                                    kernel_size=3,stride=1,padding=1),

                nn.GroupNorm(num_hiddens//2,num_hiddens//2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=num_hiddens//2, out_channels=out_channels,
                                    kernel_size=4,stride=2,padding=1),
            ]
        elif self.hierarchy_layer == 3:
            block = [ # HVQVAE Layer3
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=num_hiddens,
                                    kernel_size=3,stride=1,padding=1),
                Residual(hidden_size=num_hiddens,res_size=num_residual_hiddens),
                Residual(hidden_size=num_hiddens,res_size=num_residual_hiddens),
                nn.ConvTranspose2d(in_channels=num_hiddens,out_channels=num_hiddens,
                                    kernel_size=4,stride=2,padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=num_hiddens,out_channels=num_hiddens,
                                    kernel_size=3,stride=1,padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=out_channels,
                                    kernel_size=4,stride=2,padding=1),
            ]
        
        self.block = nn.Sequential(*block)
        self.verbose = verbose

    def forward(self,inputs):
        x_ = self.block(inputs)

        if self.verbose:
            ConsoleLogger.status('[FEATURES_DEC] input size: {}'.format(inputs.size()))
            ConsoleLogger.status('[FEATURES_DEC] decoded size at layer{}: {}'.format(self.hierarchy_layer,x_.size()))

        return x_

