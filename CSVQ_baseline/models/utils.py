import torchaudio
import torch

from config import cfg
import os

class Real2Complex:
    def __call__(self, x):
        return torch.view_as_complex(x)

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

os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['cuda_device'])
ISTFT = torchaudio.transforms.InverseSpectrogram(win_length=int(cfg['win_length']*cfg['sr']*1e-3),
                                            hop_length=int(cfg['hop_length']*cfg['sr']*1e-3)).cuda()
MERGE = Real2Complex()
denormalize = UnStandardize(stats=cfg['stats'][cfg['data_name']])

def reconstruct_audio(X): 
    ''' X: batch of STFT feature (N, 2, F, T) 
        returns: batch of recon_audios (N, L)
    '''
    X = denormalize(X.permute(0,2,3,1).contiguous()) # (N, F, T, 2)
    X = MERGE(X) # (N, F, T)

    return ISTFT(X) 
    


