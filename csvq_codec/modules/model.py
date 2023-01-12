import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.vqvae_quantizer import EMAQuantizer
from modules.tf_net import TCM_Module, CausalConv2d, CausalConvTranspose2d

from config import cfg


class CSVQ_Encoder(nn.Module):
    def __init__(self, in_channels=2, ch_mult=(4,4,8,8,16,16),num_res_block=6,num_GRU_layers=2):
        super().__init__()
        self.B = len(ch_mult)

        self.down = nn.ModuleList()
        in_ch_mult = (1,)+tuple(ch_mult)

        for i in range(self.B):
            block_in, block_out = in_channels*in_ch_mult[i], in_channels*ch_mult[i]
            self.down.append(F_downsample(block_in,block_out))
        
        # self.TCM = TCM_Module(C=ch_mult[-1]*2, F=cfg['raw_F']//2**self.B, C_up_rate=2, 
                                # res_block=num_res_block, gru_layers=num_GRU_layers)

    def forward(self,input):
        "x_shape: B, C=2, F, T"

        hs = [self.down[0](input)]

        for i in range(1,len(self.down)):
            hs.append(self.down[i](hs[-1]))

        # hs[-1] = self.TCM(hs[-1])

        return hs



class CSVQ_Decoder(nn.Module):
    def __init__(self, in_channels=2, ch_mult=(4,4,8,8,16,16), Groups=6, F=201, num_res_block=6, num_GRU_layers=2,
                    codebook_size=1024, vq_commit=0.25, C_0 = 128, C_down_rate=0.3):
        super().__init__()
        self.B = len(ch_mult)
        
        self.up = nn.ModuleList()
        out_ch_mult = (1,)+tuple(ch_mult)
        for i in reversed(range(self.B)):
            block_in, block_out = in_channels*ch_mult[i], in_channels*out_ch_mult[i]
            self.up.append(F_upsample(block_in,block_out))

        channel = (101, 51, 26, 13, 7)

        self.quantizer0 = Quantization(G=Groups, ch=ch_mult[-1]*in_channels, F=channel[self.B-1], 
                                        C_=make_nearest(C_0, Groups), codebook_size=codebook_size, vq_commit=vq_commit)

        # self.TCM = nn.Sequential(*[
        #     TCM_Module(C=ch_mult[-1]*2, F=cfg['raw_F']//2**self.B, C_up_rate=2,res_block=num_res_block, gru_layers=num_GRU_layers),
        #     TCM_Module(C=ch_mult[-1]*2, F=cfg['raw_F']//2**self.B, C_up_rate=2,res_block=num_res_block, gru_layers=num_GRU_layers)
        # ])

        self.fuse_module = nn.ModuleList([CSVQ_Fuse(G=Groups, C=ch_mult[self.B-i-1]*in_channels, F_= channel[self.B-i-1], 
                                    C_down_rate=0.3,codebook_size=codebook_size, vq_commit=vq_commit) for i in range(self.B)])
        
    def forward(self, input, Bs):
        ''' 
          input_shape: [E3,E2,E1]
          Bs: int N, the target bitstream to achieve (1 <= N <= self.B + 1)
        '''
        if Bs is None: Bs = self.B + 1 # means non-scalable
        assert Bs <= self.B + 1 
        
        z_q1, vq_loss = self.quantizer0(input[-1])  # N,ch,F',T -> (N,ch,F',T)^ here constitutes 1 bitstream
        # z_q1 = self.TCM(z_q1)

        F_dec = [z_q1]
        # F_hs_merge = []

        for i in range(self.B):
            if i > Bs - 2: # No Fuse
                Di = F_dec[i]
                if Di.size(2) < input[-i-1].size(2):
                    Di = torch.nn.functional.pad(Di,(0,0,1,0)) # pad to same shape
            else: # Fuse for the 1 -> Bs-1 bitstreams
                Di, vq_lossi = self.fuse_module[i](input[-i-1],F_dec[i]) 
                vq_loss += vq_lossi
    
            z_qi = self.up[i](Di)   # upsample: N,C',F'*2,T 
            F_dec.append(z_qi)

            # print(z_qi.shape)

        return F_dec[-1], vq_loss

class CSVQ_Fuse(nn.Module):
    def __init__(self, G, C, F_, C_down_rate, codebook_size, vq_commit):
        super().__init__()
        assert C_down_rate <= 0.5
        self.G  = G

        self.C = C

        self.C_ = int(self.C * F_ * C_down_rate)
        if self.C_ % self.G != 0:
            self.C_ = make_nearest(self.C_, self.G) # this is to make sure C_ is a multiple of G
        
        self.quantizer = Quantization(G=G, ch=self.C, F=F_, C_=self.C_, codebook_size=codebook_size, vq_commit=vq_commit)

        self.conv_block1 = nn.Sequential(*[nn.Conv2d(2*self.C,self.C,1,1), nn.Conv2d(self.C,self.C,1,1)])
        self.conv_block2 = nn.Sequential(*[nn.Conv2d(2*self.C,self.C,1,1), nn.Conv2d(self.C,self.C,1,1)])

    def forward(self, z_q, z_e):
        if z_q.size(2) < z_e.size(2):
            z_q = torch.nn.functional.pad(z_q,(0,0,1,0))
        elif z_q.size(2) > z_e.size(2):
            z_e = torch.nn.functional.pad(z_e,(0,0,1,0))
        '''
        input: 
        z_q: decoded feature (N, C, F, T) 
        z_e: encoded feature (N, C, F, T)
        returns:
        z_q_: refined decoded feature (N, C, F, T)
        '''

        # 1. concat
        z_merge = torch.cat((z_q,z_e), dim=1)  # (N, 2C, F, T)
        # 2. convolution
        
        z_merge = self.conv_block1(z_merge)    # (N, C, F, T)

        q_merge, vq_loss = self.quantizer(z_merge) # (N, C, F, T)

        # 3. concat
        z_q_ = torch.cat((z_q,q_merge), dim=1) # (N, 2C, F, T)
        # 4. convolution
        z_q_ = self.conv_block2(z_q_) # (N, C, F, T)

        return z_q_, vq_loss

class Quantization(nn.Module):
    def __init__(self, G, ch, F, C_, codebook_size, vq_commit):
        super().__init__()
        '''
        G: num of groups
        ch: input channel
        F: input frequency
        C_: fixed channel 
        '''
        self.ch = ch 
        self.down_channel_conv = nn.Conv1d(ch*F, C_, 1, 1)
        self.up_channel_conv = nn.Conv1d(C_, ch*F, 1, 1)

        K  = 4 * C_ // G
        self.quantizer = Group_Quantization(Groups=G, K=K, codebook_size=codebook_size, vq_commit=vq_commit)
    
    def forward(self, x):
        '''x:  (N, ch, F, T)'''
        assert self.ch == x.size(1)
        
        x = fold(x) # (N, ch*F, T)

        # 1. downsample to fix channel
        x = self.down_channel_conv(x) # (N, C_, T)
        
        # 2. vq
        q, vq_loss = self.quantizer(x) # (N, C_, T)

        # 3. upsample to initial channel
        q = self.up_channel_conv(q) # (N, ch*F, T)

        q = unfold(q,self.ch) # (N, ch, F, T)

        return q, vq_loss


class Group_Quantization(nn.Module):
    def __init__(self, Groups=3, K=40, codebook_size=1024, vq_commit=0.25):
        super().__init__()
        self.K = K
        self.G = Groups

        self.quantizer = nn.ModuleList([EMAQuantizer(self.K, codebook_size, vq_commit) for _ in range(self.G)])
            
    def forward(self, z):
        '''
        z: merge feature (N, C_, T)
        q: quantized feature (N, C_, T)^
        C_ == K * G
        '''
        z = torch.reshape(z, (z.size(0), z.size(1)*4, -1)) # (N, 4*C_, T//4)
        assert z.size(1) == self.K * self.G

        q_merge, vq_loss = [], torch.tensor(0.0, device=cfg['device'])
        
        for i in range(self.G):
            q_i, vq_loss, _, ppl = self.quantizer[i](z[:,self.K*i:self.K*(i+1),:])
            q_merge.append(q_i)

        q_merge = torch.cat(q_merge,dim=1) # (N, 4*C_, T//4)
        q_merge = torch.reshape(q_merge,(z.size(0),z.size(1)//4,-1)) # (N, C_, T)

        return q_merge, vq_loss


class F_downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # self.down = nn.Conv2d(in_channels, out_channels, kernel_size=(2,1), stride=(2,1))

        self.down = CausalConv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=3, stride=(2,1), padding=(1,1), 
                                mask_type='B', data_channels=in_channels)

        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        "x_shape: B, C=2, F, T"
        x = self.down(x)
        x = self.norm(x)
        x = self.act(x)
        "x_shape: B, C=2, F//2, T"

        return x

class F_upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2,1), stride=(2,1))

        self.up = CausalConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=3, stride=(2,1), padding=(1,1), 
                                mask_type='B', data_channels=in_channels)

        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        "x_shape: B, C=2, F//2, T"
        x = self.up(x)
        x = self.norm(x)
        x = self.act(x)
        "x_shape: B, C=2, F, T"
        
        return x


def fold(te):
    '''
    te: N, C, F', T
    return: N, C*F', T
    '''
    return torch.reshape(te, (te.size(0), -1, te.size(3)))

def unfold(te, C):
    '''
    te: N, C*F', T
    return: N, C, F', T
    '''
    return torch.reshape(te, (te.size(0), C, -1, te.size(2)))

def make_nearest(C, g):
    for i in range(C-g, C):
        if i % g == 0: return i

# from torch.nn.modules.utils import _pair
# class CausalConv2d(nn.Conv2d):

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True):
#         kernel_size = _pair(kernel_size)
#         stride = _pair(stride)
#         dilation = _pair(dilation)
#         if padding is None:
#             padding = [int((kernel_size[i] -1) * dilation[i]) for i in range(len(kernel_size))]
#         self.left_padding = _pair(padding)
#         super().__init__(in_channels, out_channels, kernel_size,
#                                            stride=stride, padding=0, dilation=dilation,
#                                            groups=groups, bias=bias)
#     def forward(self, inputs):
#         inputs = F.pad(inputs, (self.left_padding[1], 0, self.left_padding[0], 0))
#         output = super().forward(inputs)
#         return output



