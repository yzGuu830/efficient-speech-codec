import torch.nn as nn
import torch.nn.functional as F

import torchaudio
import torch

class L2Loss(nn.Module):
    def __init__(self, weight=1.0, power_law=True):
        super().__init__() 
        self.power_law = power_law
        self.weight = weight

    def forward(self, raw_feat, recon_feat):
        assert raw_feat.size() == recon_feat.size() #(N,2,F,T)
        if self.power_law:
            raw_feat = power_law(raw_feat, power=0.3, eps=1e-10)
            recon_feat = power_law(recon_feat, power=0.3, eps=1e-10)

        return self.weight*F.mse_loss(raw_feat,recon_feat,reduction="none").mean([1,2,3])

def power_law(stft, power=0.3, eps=1e-10):
    mask = torch.sign(stft)
    power_law_compressed = (torch.abs(stft) + eps) ** power
    power_law_compressed = power_law_compressed * mask
    return power_law_compressed

class MelLoss(nn.Module):
    def __init__(self, 
                 weight=1.0,
                 win_lengths=[32,64,128,256,512,1024,2048], 
                 n_mels=[5,10,20,40,80,160,320], 
                 clamp_eps=1e-5,):
        super().__init__()

        self.n_mels = n_mels
        self.mel_transf = nn.ModuleList( [
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=w, win_length=w, 
                hop_length=w//4, n_mels=n_mels[i], power=1)
            for i, w in enumerate(win_lengths)
        ] )
        self.clamp_eps = clamp_eps
        self.weight = weight

    def forward(self, raw_audio, recon_audio):
        assert raw_audio.size() == recon_audio.size()

        mel_loss = 0.0
        for mel_trans in self.mel_transf:
            x_mels, y_mels = mel_trans(raw_audio), mel_trans(recon_audio)

            mel_loss += F.l1_loss(x_mels, y_mels, reduction="none").mean([1,2]) # magnitude loss
            mel_loss += F.l1_loss(  # log mel loss
                x_mels.clamp(self.clamp_eps).pow(2).log10(),
                y_mels.clamp(self.clamp_eps).pow(2).log10(), 
                reduction="none"
            ).mean([1,2])

        return self.weight*mel_loss