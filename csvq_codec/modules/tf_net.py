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



class TCM_ResBlock(nn.Module):
    '''
    Dilated temporal convolution module (TCM) 
    '''
    def __init__(self, C=32, F=3, C_up_rate=2, dilation_rate=1):
        super().__init__()
        self.ch = C*F
        self.C_up = self.ch * C_up_rate
                
        self.d_conv = nn.Conv1d(self.C_up, self.C_up, 3, padding=dilation_rate, dilation=dilation_rate, groups=self.C_up)
                
        self.up_channel_conv = nn.Conv1d(self.ch, self.C_up, 1, 1)
        self.down_channel_conv = nn.Conv1d(self.C_up, self.ch, 1, 1)

        self.act = nn.ReLU(inplace=True)
        
        self.up_norm = nn.BatchNorm1d(self.C_up)
        self.down_norm = nn.BatchNorm1d(self.ch)


    def forward(self,x):
        "x_shape: B, C, F, T"
        # print("x_shape: ", x.shape)
        x_ = torch.reshape(x,(x.size(0),-1,x.size(3))) # (B, C*F, T)
        assert x_.size(1) == self.ch

        x_ = self.up_channel_conv(x_) # (B, C_up, T)
        x_ = self.up_norm(self.act(x_))

        # print("up_x: ", x_.shape)
        filter_x = self.d_conv(x_) # (B, C_up, T)^
        # print("filter_x: ", filter_x.shape)

        filter_x = self.down_channel_conv(filter_x) # (B, C*F, T)^
        filter_x = self.down_norm(self.act(filter_x))

        # print("down_x: ", filter_x.shape)

        filter_x  = torch.reshape(filter_x, x.size()) # (B, C, F, T)

        return filter_x + x # residual

class TCM_Module(nn.Module):
    def __init__(self, C=32, F=3, C_up_rate=2, res_block=6, gru_layers=2):
        super().__init__()

        TCM_Block = [TCM_ResBlock(C, F, C_up_rate, dilation_rate=2**i) for i in range(res_block)]
        self.TCM_Block = nn.Sequential(*TCM_Block)

        self.num_gru_layers = gru_layers
        self.ch = C*F
        
        self.rnn = nn.GRU(input_size=self.ch,hidden_size=self.ch,num_layers=self.num_gru_layers,batch_first=True)

    def forward(self, x):
        "x_shape: B, C, F, T"
        x = self.TCM_Block(x)

        x_ = torch.reshape(x, (x.size(0),x.size(3),-1)) # (B, T, C*F)

        h_0 = torch.randn(self.num_gru_layers,x.size(0),self.ch, device=cfg['device'])
        x_, h_0 = self.rnn(x_, h_0) # (B, T, C*F)

        x_ = torch.reshape(x_,x.size())

        return x_