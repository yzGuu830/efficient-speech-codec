import torch
from config import cfg
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

class UnStandardize:
    def __init__(self, stats):
        self.m = stats[0]
        self.s = stats[1]
    
    def __call__(self, input):
        broadcast_size = [1,2]
        m, s = torch.tensor(self.m, dtype=input.dtype).view(broadcast_size).to(input.device), \
           torch.tensor(self.s, dtype=input.dtype).view(broadcast_size).to(input.device)
        input = input.mul(s).add(m)
        return input

denormalize = UnStandardize(stats=cfg['stats'][cfg['data_name']])

def feat2spec(X): 
    ''' 
        convert normalized feature back to stft spectrogram
        X: batch of STFT feature (N, 2, F, T) 
        returns: batch of complex recon features (N, F, T)
    '''

    X = denormalize(X.permute(0,2,3,1).contiguous()) # (N, F, T, 2)
    X = torch.view_as_complex(X) # (N, F, T) real2complex

    return X

def power_law_compress(data, power=0.6):
    # power_law compressed audio signal
    mask = torch.ones(data.shape, device=cfg['device'])
    mask = mask.masked_fill((data<0),-1)

    data = torch.pow(torch.abs(data),power)
    data = data * mask
    return data


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

def find_nearest(C, g):
    for i in range(C-g, C):
        if i % g == 0: return i
    


