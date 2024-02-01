import torch.nn as nn
import torch, torchaudio
import torch.nn.functional as F
from pesq import pesq


class MelDistance(nn.Module):
    def __init__(self, 
                 win_lengths=[32, 64, 128, 256, 512, 1024, 2048], 
                 n_mels=[5, 10, 20, 40, 80, 160, 320], 
                 clamp_eps=1e-5,):
        super().__init__()

        self.n_mels = n_mels
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

            mel_loss += F.l1_loss(  # log mel loss
                x_mels.clamp(self.clamp_eps).pow(2).log10(),
                y_mels.clamp(self.clamp_eps).pow(2).log10(), 
                reduction="none"
            ).mean(dim=[1,2])

        return mel_loss

class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals or aligned features.

    Parameters
    ----------
    scaling : int, optional
        Whether to use scale-invariant (True) or
        signal-to-noise ratio (False), by default True
    reduction : str, optional
        How to reduce across the batch (either 'mean',
        'sum', or none).], by default ' mean'
    zero_mean : int, optional
        Zero mean the references and estimates before
        computing the loss, by default True
    clip_min : int, optional
        The minimum possible loss value. Helps network
        to not focus on making already good examples better, by default None
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(
        self,
        scaling: int = True,
        reduction: str = "none",
        zero_mean: int = True,
        clip_min: int = None,
        weight: float = 1.0,
    ):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = weight
        super().__init__()

    def forward(self, x, y):
        eps = 1e-8
        # nb, nc, nt

        references = x.unsqueeze(1) if x.dim() == 2 else x
        estimates = y.unsqueeze(1) if y.dim() == 2 else y

        nb = references.shape[0]
        references = references.reshape(nb, 1, -1).permute(0, 2, 1)
        estimates = estimates.reshape(nb, 1, -1).permute(0, 2, 1)

        # samples now on axis 1
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0

        _references = references - mean_reference
        _estimates = estimates - mean_estimate

        references_projection = (_references**2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling
            else 1
        )

        e_true = scale * _references
        e_res = _estimates - e_true

        signal = (e_true**2).sum(dim=1)
        noise = (e_res**2).sum(dim=1)
        sdr = -10 * torch.log10(signal / noise + eps)

        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)

        return sdr.squeeze(1)

class PESQ:      
    def __init__(self, sample_rate, device="cpu") -> None:
        
        self.sr = sample_rate
        self.default_sr = 16000
        
        self.resampler = Resampler(sample_rate, self.default_sr, device=device) if (self.sr != self.default_sr) else None

    def __call__(self, x, y):
        """
            x: source audio Tensor [bs, T]
            y: recon audio Tensor [bs, T]
        """
        bs = x.size(0)

        if self.resampler is not None:
            x = self.resampler(x)
            y = self.resampler(y)

        batch_pesq = []
        for b in range(bs):
            ref = x[b].cpu().numpy()
            deg = y[b].cpu().numpy()
            batch_pesq.append(pesq(self.default_sr, ref, deg, 'wb'))
        
        return batch_pesq
        
class ViSQOL:
    def __init__(self) -> None:
        pass

class STFTDistance(nn.Module):

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for s in self.stft_params:
            x.stft(s.window_length, s.hop_length, s.window_type)
            y.stft(s.window_length, s.hop_length, s.window_type)
            loss += self.log_weight * self.loss_fn(
                x.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                y.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x.magnitude, y.magnitude)
        return loss

class Resampler:
    def __init__(self, sample_rate, 
                resample_rate=16000,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
                device="cpu") -> None:
        
        self.torchaudio_resampler = torchaudio.transforms.Resample(
                    sample_rate,
                    resample_rate,
                    lowpass_filter_width=lowpass_filter_width,
                    rolloff=rolloff,
                    resampling_method=resampling_method,
                    beta=beta,
                ).to(device)
        
    def __call__(self, x):

        return self.torchaudio_resampler(x)


if __name__ == "__main__":
    mel_distance_metric = MelDistance(
            win_lengths=[32,64,128,256,512,1024,2048], 
            n_mels=[5,10,20,40,80,160,320], 
            clamp_eps=1e-5
    )

    sisdr_metric = SISDRLoss(
        scaling = True,
        reduction = "none",
        zero_mean = True,
        clip_min = None,
        weight = 1.0,
    )

    pesq_metric = PESQ(sample_rate=24000)

    sr = 24000
    torch.manual_seed(0)
    x = torch.randn(4, sr*10)

    y1 = torch.ones(4, sr*10)
    y2 = torch.randn(4, sr*10)


    print("mel distance: y1=\n{} y2=\n{}".format(
        mel_distance_metric(x, y1), mel_distance_metric(x, y2)
    ))

    print("sisdr: y1=\n{} y2=\n{}".format(
        sisdr_metric(x, y1), sisdr_metric(x, y2)
    ))

    print("pesq: y1=\n{} y2=\n{}".format(
        pesq_metric(x, y1), pesq_metric(x, y2)
    ))