import torch.nn as nn
import torch.nn.functional as F
import torch

import torchaudio
import config as cfg


class MSE_LOSS(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, raw_feat, recon_feat):
        assert raw_feat.size() == recon_feat.size() #(N,2,F,T)

        return F.mse_loss(raw_feat,recon_feat, reduction='none').mean(dim=[1,2,3]) if cfg.num_workers > 1 \
            else F.mse_loss(raw_feat,recon_feat)
        # [bs, loss]

class MEL_LOSS(nn.Module):
    def __init__(self, n_fft=2048, n_mels=64):
        super().__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels # https://proceedings.neurips.cc/paper/2020/file/9873eaad153c6c960616c89e54fe155a-Paper.pdf
        
        # 64 -> 2048
        self.mel_transf1 = torchaudio.transforms.MelSpectrogram(n_fft=400,win_length=2**6, hop_length=2**4, n_mels=self.n_mels,power=1)
        self.mel_transf2 = torchaudio.transforms.MelSpectrogram(n_fft=400,win_length=2**7, hop_length=2**5, n_mels=self.n_mels,power=1)
        self.mel_transf3 = torchaudio.transforms.MelSpectrogram(n_fft=400,win_length=2**8, hop_length=2**6, n_mels=self.n_mels,power=1)
        self.mel_transf4 = torchaudio.transforms.MelSpectrogram(n_fft=600,win_length=2**9, hop_length=2**7, n_mels=self.n_mels,power=1)
        self.mel_transf5 = torchaudio.transforms.MelSpectrogram(n_fft=1024,win_length=2**10, hop_length=2**8, n_mels=self.n_mels,power=1)
        self.mel_transf6 = torchaudio.transforms.MelSpectrogram(n_fft=2048,win_length=2**11, hop_length=2**9, n_mels=self.n_mels,power=1)

    def forward(self, raw_audio, recon_audio):
        assert raw_audio.size() == recon_audio.size()

        loss_1 = (self.mel_transf1(raw_audio) - self.mel_transf1(recon_audio)).abs().mean(dim=[1,2])
        loss_2 = (self.mel_transf2(raw_audio) - self.mel_transf2(recon_audio)).abs().mean(dim=[1,2])
        loss_3 = (self.mel_transf3(raw_audio) - self.mel_transf3(recon_audio)).abs().mean(dim=[1,2])
        loss_4 = (self.mel_transf4(raw_audio) - self.mel_transf4(recon_audio)).abs().mean(dim=[1,2])
        loss_5 = (self.mel_transf5(raw_audio) - self.mel_transf5(recon_audio)).abs().mean(dim=[1,2])
        loss_6 = (self.mel_transf6(raw_audio) - self.mel_transf6(recon_audio)).abs().mean(dim=[1,2])

        mel_loss = (loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6)

        return mel_loss # [bs, loss]
    

class MS_LOSS(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, enc_hs, dec_hs, num_stream):
        assert len(enc_hs) == len(dec_hs)

        # for hs in enc_hs:
        #     print(hs.shape)
        # print("______________")
        # for hs in dec_hs:
        #     print(hs.shape)

        multi_scale_loss = torch.zeros(enc_hs[-1].size(0), device=enc_hs[-1].device) if cfg.num_workers > 1 \
                else torch.tensor(0.0, device=enc_hs[-1].device)
        
        for i in range(num_stream):
            multi_scale_loss += F.mse_loss(dec_hs[i], enc_hs[-1-i], reduction='none').mean(dim=[1,2]) if cfg.num_workers > 1 \
                            else F.mse_loss(dec_hs[i], enc_hs[-1-i]) 
        
        return multi_scale_loss
