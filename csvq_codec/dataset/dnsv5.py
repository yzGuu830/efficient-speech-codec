from torch.utils.data import Dataset
import torchaudio
import torch

import os
from utils import load

from config import cfg


class DNSV5(Dataset):
    data_name = 'DNS_CHALLENGE'

    def __init__(self, root, split, transform=None) -> None:
        self.root = os.path.expanduser(root)
        self.split = split

        self.transform = transform

        self.stft = torchaudio.transforms.Spectrogram(win_length=int(cfg['win_length']*cfg['sr']*1e-3),hop_length=int(cfg['hop_length']*cfg['sr']*1e-3),power=None)

        self.source_audio = load('/hpc/group/tarokhlab/yg172/data/DNS_CHALLENGE/processed/{}.pt'.format(self.split), mode='torch')

    def __len__(self): # return dataset length
        return self.source_audio.size(0)

    def __getitem__(self, idx): # return ith audio waveform data 
        input = {'audio':self.source_audio[idx][:-80]}
    
        input['stft_feat'] = torch.view_as_real(self.stft(input['audio'])) # [1, F, T, 2]

        if self.transform:
            input['stft_feat'] = self.transform(input['stft_feat'])

        return input

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

