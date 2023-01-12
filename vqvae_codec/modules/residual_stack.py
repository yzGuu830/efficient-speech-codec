from modules.residual import Residual

import torch.nn as nn
import torch.nn.functional as F


class ResidualStack(nn.Module):

    def __init__(self, hidden_size, num_residual_layers, res_size):
        super(ResidualStack, self).__init__()
        
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [Residual(hidden_size, res_size)] * self._num_residual_layers)
        
    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        x = F.relu(x)
        return x