import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
from config import cfg

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

class MSE_Loss(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, raw_feat, recon_feat):
        assert raw_feat.size() == recon_feat.size() #(N,2,F,T)

        return F.mse_loss(raw_feat,recon_feat, reduction='none').mean(dim=[1,2,3]) # [bs, loss]

        # return F.mse_loss(raw_feat,recon_feat)

class Mel_Loss(nn.Module):
    def __init__(self, n_fft=2048, n_mels=128):
        super().__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels # https://proceedings.neurips.cc/paper/2020/file/9873eaad153c6c960616c89e54fe155a-Paper.pdf
        
        # 64 -> 2048
        self.mel_transf1 = torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft,win_length=2**6, hop_length=2**4, n_mels=self.n_mels,power=1)
        self.mel_transf2 = torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft,win_length=2**7, hop_length=2**5, n_mels=self.n_mels,power=1)
        self.mel_transf3 = torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft,win_length=2**8, hop_length=2**6, n_mels=self.n_mels,power=1)
        self.mel_transf4 = torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft,win_length=2**9, hop_length=2**7, n_mels=self.n_mels,power=1)
        self.mel_transf5 = torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft,win_length=2**10, hop_length=2**8, n_mels=self.n_mels,power=1)
        self.mel_transf6 = torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft,win_length=2**11, hop_length=2**9, n_mels=self.n_mels,power=1)

    def forward(self, raw_audio, recon_audio):
        assert raw_audio.size() == recon_audio.size()
        
        # loss_1 = F.l1_loss(self.mel_transf1(raw_audio), self.mel_transf1(recon_audio) , reduction='none')
        # loss_2 = F.l1_loss(self.mel_transf2(raw_audio), self.mel_transf2(recon_audio) , reduction='none')
        # loss_3 = F.l1_loss(self.mel_transf3(raw_audio), self.mel_transf3(recon_audio) , reduction='none')
        # loss_4 = F.l1_loss(self.mel_transf4(raw_audio), self.mel_transf4(recon_audio) , reduction='none')
        # loss_5 = F.l1_loss(self.mel_transf5(raw_audio), self.mel_transf5(recon_audio) , reduction='none')
        # loss_6 = F.l1_loss(self.mel_transf6(raw_audio), self.mel_transf6(recon_audio) , reduction='none')

        # mel_loss = (loss_1.mean(dim=[1,2]) + loss_2.mean(dim=[1,2]) + loss_3.mean(dim=[1,2]) 
        #             + loss_4.mean(dim=[1,2]) + loss_5.mean(dim=[1,2]) + loss_6.mean(dim=[1,2]))/6


        loss_1 = (self.mel_transf1(raw_audio) - self.mel_transf1(recon_audio)).abs().mean(dim=[1,2])
        loss_2 = (self.mel_transf2(raw_audio) - self.mel_transf2(recon_audio)).abs().mean(dim=[1,2])
        loss_3 = (self.mel_transf3(raw_audio) - self.mel_transf3(recon_audio)).abs().mean(dim=[1,2])
        loss_4 = (self.mel_transf4(raw_audio) - self.mel_transf4(recon_audio)).abs().mean(dim=[1,2])
        loss_5 = (self.mel_transf5(raw_audio) - self.mel_transf5(recon_audio)).abs().mean(dim=[1,2])
        loss_6 = (self.mel_transf6(raw_audio) - self.mel_transf6(recon_audio)).abs().mean(dim=[1,2])

        mel_loss = (loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6)

        return mel_loss # [bs, loss]




# def fast_stft(data,power=False,**kwargs):
#     # directly transform the wav to the input
#     # power law = A**0.3 , to prevent loud audio from overwhelming soft audio
#     if power:
#         data = power_law(data)
#     return real_imag_expand(stft(data))

# def fast_istft(F,power=False,**kwargs):
#     # directly transform the frequency domain data to time domain data
#     # apply power law
#     T = istft(real_imag_shrink(F))
#     if power:
#         T = power_law(T,(1.0/0.6))
#     return T