import torch.nn as nn
import torch.nn.functional as F

import torchaudio.transforms as T
import torch

MEL_WINDOWS = [32,64,128,256,512,1024,2048]
MEL_BINS = [5,10,20,40,80,160,320]
SR = 16000
POWER = 0.3

class ComplexSTFTLoss(nn.Module):
    """L2 Loss on Complex STFTs (Power Law Compressed https://arxiv.org/pdf/1811.07030)"""
    def __init__(self, weight=1.0, power_law=True):
        super().__init__() 
        self.power_law = power_law 
        self.weight = weight

    def forward(self, raw_feat, recon_feat):
        """
        Args: 
            raw_feat/recon_feat: (B,2,F,T)
            returns: (B,)
        """
        if self.power_law:
            raw_feat = power_law(raw_feat, power=POWER)
            recon_feat = power_law(recon_feat, power=POWER)

        return self.weight * F.mse_loss(raw_feat,recon_feat,reduction="none").mean([1,2,3])

def power_law(stft, power=POWER, eps=1e-10):
    mask = torch.sign(stft)
    power_law_compressed = (torch.abs(stft) + eps) ** power
    power_law_compressed = power_law_compressed * mask
    return power_law_compressed

class MelSpectrogramLoss(nn.Module):
    """
    L1 MelSpectrogram Loss 
    Implementation adapted from https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py
    """
    def __init__(self, weight=1.0,
                 win_lengths=MEL_WINDOWS, n_mels=MEL_BINS, clamp_eps=1e-5,):
        super().__init__()

        self.n_mels = n_mels
        self.mel_transf = nn.ModuleList( [
            T.MelSpectrogram(
                sample_rate=SR, n_fft=w, win_length=w, 
                hop_length=w//4, n_mels=n_mels[i], power=1)
            for i, w in enumerate(win_lengths)
        ] )
        self.clamp_eps = clamp_eps
        self.weight = weight

    def forward(self, raw_audio, recon_audio):
        """
        Args: 
            raw_audio/recon_audio: (B,L)
            returns: (B,)
        """
        mel_loss = 0.0
        for mel_trans in self.mel_transf:
            x_mels, y_mels = mel_trans(raw_audio), mel_trans(recon_audio)

            # magnitude loss
            mel_loss += F.l1_loss(x_mels, y_mels, reduction="none").mean([1,2]) 
            # log magnitude loss
            mel_loss += F.l1_loss(  
                x_mels.clamp(self.clamp_eps).pow(2).log10(),
                y_mels.clamp(self.clamp_eps).pow(2).log10(), 
                reduction="none"
            ).mean([1,2])

        return self.weight * mel_loss