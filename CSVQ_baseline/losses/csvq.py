import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
from config import cfg

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['cuda_device'])

class Compress_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, raw_feat, recon_feat):
        assert raw_feat.size() == recon_feat.size()

        return F.mse_loss(raw_feat,recon_feat)

class Mel_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, raw_audio, recon_audio):
        assert raw_audio.size() == recon_audio.size()
        
        mel_loss = torch.tensor(0.0,device=cfg['device'])

        for i in range(6,12): # 64 -> 2048
            mel_transf = torchaudio.transforms.MelSpectrogram(n_fft=2048,win_length=2**i,n_mels=32).cuda()
            mel_loss += torch.mean(torch.abs(mel_transf(raw_audio) - mel_transf(recon_audio)))

        mel_loss = mel_loss / 6

        return mel_loss
