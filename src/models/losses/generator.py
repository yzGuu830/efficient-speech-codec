import torch.nn as nn
import torch.nn.functional as F

import torchaudio
import torch

class L2Loss(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, raw_feat, recon_feat):
        assert raw_feat.size() == recon_feat.size() #(N,2,F,T)
        return F.mse_loss(raw_feat,recon_feat,reduction="none").mean([1,2,3])

class MELLoss(nn.Module):
    def __init__(self, 
                 win_lengths=[32, 64, 128, 256, 512, 1024, 2048], 
                 n_mels=[5, 10, 20, 40, 80, 160, 320], 
                 clamp_eps=1e-5,):
        super().__init__()

        self.n_mels = n_mels # https://proceedings.neurips.cc/paper/2020/file/9873eaad153c6c960616c89e54fe155a-Paper.pdf
        self.mel_transf = nn.ModuleList( [
            torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                 n_fft=w,
                                                 win_length=w, 
                                                 hop_length=w//4, 
                                                 n_mels=n_mels[i],
                                                 power=1)
            for i, w in enumerate(win_lengths)
        ] )

        self.clamp_eps = clamp_eps

    def forward(self, raw_audio, recon_audio):
        assert raw_audio.size() == recon_audio.size()

        mel_loss = 0.0
        for mel_trans in self.mel_transf:
            x_mels, y_mels = mel_trans(raw_audio), mel_trans(recon_audio)

            mel_loss += F.l1_loss(x_mels, y_mels, reduction="none").mean([1,2]) # megnitude loss

            mel_loss += F.l1_loss(  # log mel loss
                x_mels.clamp(self.clamp_eps).pow(2).log10(),
                y_mels.clamp(self.clamp_eps).pow(2).log10(), 
                reduction="none"
            ).mean([1,2])

        return mel_loss
    

if __name__ == "__main__":
    x = torch.randn(1, 48000)
    y = torch.randn(1,48000)
    mel_loss = MELLoss()

    print(mel_loss(x, y))