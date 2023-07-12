from torch.utils.data import Dataset
import torch
import torchaudio
import os
from utils import load
import config as cfg

class DNSV5(Dataset):
    data_name = 'DNS_CHALLENGE'
    """https://github.com/microsoft/DNS-Challenge"""

    def __init__(self, root, split, transform=None) -> None:
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform

        nfft = (cfg.init_H - 1) * 2                        
        self.feat_trans = torchaudio.transforms.Spectrogram(n_fft=nfft,
                            win_length=int(cfg.win_length*cfg.sr*1e-3),
                            hop_length=int(cfg.hop_length*cfg.sr*1e-3), power=None)

        self.source_audio = load('{}/DNS_CHALLENGE/processed/{}.pt'.format(cfg.data_path, self.split), mode='torch')

    def __len__(self): # return dataset length
        return self.source_audio.size(0)

    def __getitem__(self, idx): # return ith audio waveform data 
        input = {'audio':self.source_audio[idx][:-80]}
        input['feat'] = torch.view_as_real(self.feat_trans(input['audio'])) # [F=nfft//2-1, T=600, 2]

        if self.transform:
            input['feat'] = self.transform(input['feat'])

        return input

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str