import torch.nn as nn
import torch.nn.functional as F

import torchaudio
import torch

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, raw_feat, recon_feat):
        assert raw_feat.size() == recon_feat.size() #(N,2,F,T)
        return F.mse_loss(raw_feat,recon_feat)

class TimeLoss(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
    
    def forward(self, raw_audio, recon_audio):
        assert raw_audio.size() == recon_audio.size() #(N,L)
        return F.mse_loss(raw_audio,recon_audio)
    
class FreqLoss(nn.Module):
    def __init__(self, n_mels=64) -> None:
        super().__init__()
        
        self.stfts = nn.ModuleList([
            torchaudio.transforms.Spectrogram(n_fft=2**i, hop_length=2**(i-2), win_length=2**i, power=1)
            for i in range(5, 12)
        ])

        self.mel_trans = nn.ModuleList([
            torchaudio.transforms.MelScale(n_mels=n_mels, 
                                           sample_rate=16000, 
                                           n_stft=(2**i)//2+1)
            for i in range(5, 12)
        ])

    def forward(self, raw_audio, recon_audio):
        assert raw_audio.size() == recon_audio.size() #(N,L)

        distance = torch.tensor(0.0, device=raw_audio.device)

        for i, stft in enumerate(self.stfts):
            mag_spec = stft(raw_audio)
            mag_spec_recon = stft(recon_audio)

            # mean = mag_spec.mean(dim=[1, 2], keepdim=True)
            # std = mag_spec.std(dim=[1, 2], keepdim=True)
            # mag_spec_norm = mag_spec.sub(mean).div(std)

            # mean = mag_spec_recon.mean(dim=[1, 2], keepdim=True)
            # std = mag_spec_recon.std(dim=[1, 2], keepdim=True)
            # mag_spec_recon_norm = mag_spec_recon.sub(mean).div(std)

            mel_raw = self.mel_trans[i](mag_spec)
            mel_recon = self.mel_trans[i](mag_spec_recon)

            distance += ((mel_raw-mel_recon).abs().mean() + torch.square(mel_raw-mel_recon).mean()) 
        
        return distance / len(self.stfts)


class MELLoss(nn.Module):
    def __init__(self, n_fft=2048, n_mels=64):
        super().__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels # https://proceedings.neurips.cc/paper/2020/file/9873eaad153c6c960616c89e54fe155a-Paper.pdf
        
        # 64 -> 2048
        self.mel_transf1 = torchaudio.transforms.MelSpectrogram(n_fft=2048,win_length=2**6, hop_length=2**4, n_mels=self.n_mels,power=1)
        self.mel_transf2 = torchaudio.transforms.MelSpectrogram(n_fft=2048,win_length=2**7, hop_length=2**5, n_mels=self.n_mels,power=1)
        self.mel_transf3 = torchaudio.transforms.MelSpectrogram(n_fft=2048,win_length=2**8, hop_length=2**6, n_mels=self.n_mels,power=1)
        self.mel_transf4 = torchaudio.transforms.MelSpectrogram(n_fft=2048,win_length=2**9, hop_length=2**7, n_mels=self.n_mels,power=1)
        self.mel_transf5 = torchaudio.transforms.MelSpectrogram(n_fft=2048,win_length=2**10, hop_length=2**8, n_mels=self.n_mels,power=1)
        self.mel_transf6 = torchaudio.transforms.MelSpectrogram(n_fft=2048,win_length=2**11, hop_length=2**9, n_mels=self.n_mels,power=1)

    def forward(self, raw_audio, recon_audio):
        assert raw_audio.size() == recon_audio.size()

        loss_1 = (self.mel_transf1(raw_audio) - self.mel_transf1(recon_audio)).abs().mean()
        loss_2 = (self.mel_transf2(raw_audio) - self.mel_transf2(recon_audio)).abs().mean()
        loss_3 = (self.mel_transf3(raw_audio) - self.mel_transf3(recon_audio)).abs().mean()
        loss_4 = (self.mel_transf4(raw_audio) - self.mel_transf4(recon_audio)).abs().mean()
        loss_5 = (self.mel_transf5(raw_audio) - self.mel_transf5(recon_audio)).abs().mean()
        loss_6 = (self.mel_transf6(raw_audio) - self.mel_transf6(recon_audio)).abs().mean()

        mel_loss = (loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6)
        return mel_loss
