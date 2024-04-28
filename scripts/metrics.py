import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
from pesq import pesq

MEL_WINDOWS = [32,64,128,256,512,1024,2048]
MEL_BINS = [5,10,20,40,80,160,320]
SR = 16000 

class EntropyCounter:
    """Counter maintaining codebook utilization rate on a held-out validation set"""
    def __init__(self, codebook_size=1024, 
                 num_streams=6, num_groups=3, 
                 device="cuda"):

        self.num_groups = num_groups
        self.codebook_size = codebook_size
        self.device = device

        self.reset_stats(num_streams)

    def reset_stats(self, num_streams):
        self.codebook_counts = {
                f"stream_{S}_group_{G+1}": torch.zeros(self.codebook_size, device=self.device) \
                    for S in range(num_streams) for G in range(self.num_groups)
                } # counts codeword stats for each codebook
        self.total_counts = 0
        self.dist = None    # posterior distribution for each codebook
        self.entropy = None # entropy stats for each codebook

        self.max_entropy_per_book = np.log2(self.codebook_size)
        self.max_total_entropy = num_streams * self.num_groups * self.max_entropy_per_book
        self.num_streams = num_streams

    def update(self, codes):
        """ Update codebook counts and total counts from a batch of codes
        Args:
            codes: (B, num_streams, group_size, *)
        """ 
        assert codes.size(1) == self.num_streams and codes.size(2) == self.num_groups, "code indices size not match"
        num_codes = codes.size(0) * codes.size(-1)
        self.total_counts += num_codes

        for s in range(self.num_streams):
            stream_s_code = codes[:, s]                      # (B, group_size, *)
            for g in range(self.num_groups):
                stream_s_group_g_code = stream_s_code[:,g]   # (B, *)
                one_hot = F.one_hot(stream_s_group_g_code, num_classes=self.codebook_size) # (B, *, codebook_size)
                self.codebook_counts[f"stream_{s}_group_{g+1}"] += one_hot.view(-1, self.codebook_size).sum(0) # (*, codebook_size)
        
    def _form_distribution(self):
        """After iterating over a held-out set, compute posterior distribution for each codebook"""
        assert self.total_counts > 0, "No data collected, please update on a specific dataset"
        self.dist = {}
        for k, _counts in self.codebook_counts.items():
            self.dist[k] = _counts / torch.tensor(self.total_counts, device=_counts.device)
    
    def _form_entropy(self):
        """After forming codebook posterior distributions, compute entropy for each distribution"""
        assert self.dist is not None, "Please compute posterior distribution first using self._form_distribution()"
        
        self.entropy = {}
        for k, dist in self.dist.items():
            self.entropy[k] = (-torch.sum(dist * torch.log2(dist+1e-10))).item()
            
    def compute_utilization(self):
        """After forming entropy statistics for each codebook, compute utilization ratio (bitrate efficiency)"""
        if self.dist is None: self._form_distribution()
        if self.entropy is None: self._form_entropy()
        
        utilization = {}
        for k, e in self.entropy.items():
            utilization[k] = round(e/self.max_entropy_per_book, 4)

        return round(sum(self.entropy.values())/self.max_total_entropy, 4), utilization 

class PESQ:      
    """Batch-wise computing of PESQ scores"""
    def __call__(self, x, y):
        """
        Args:
            x: source audio Tensor (B, L)
            y: recon audio Tensor  (B, L)
            returns: (B,)
        """
        batch_pesq = []
        for b in range(x.size(0)):
            ref = x[b].cpu().numpy()
            deg = y[b].cpu().numpy()
            batch_pesq.append(pesq(SR, ref, deg, 'wb'))
        
        return torch.tensor(batch_pesq)

class MelSpectrogramDistance(nn.Module):
    """
    L1 Log MelSpectrogram Distance 
    Implementation adapted from https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py
    """
    def __init__(self, win_lengths=MEL_WINDOWS, 
                 n_mels=MEL_BINS, clamp_eps=1e-5,):
        super().__init__()
        self.mel_transf = nn.ModuleList([
                T.MelSpectrogram(sample_rate=SR, 
                                n_fft=w, win_length=w, hop_length=w//4, 
                                n_mels=n_mels[i], power=1)
                for i, w in enumerate(win_lengths)
            ])
        self.clamp_eps = clamp_eps

    def forward(self, raw_audio, recon_audio):
        mel_loss = 0.0
        for mel_trans in self.mel_transf:
            x_mels, y_mels = mel_trans(raw_audio), mel_trans(recon_audio)
            mel_loss += F.l1_loss(  # log mel loss
                x_mels.clamp(self.clamp_eps).pow(2).log10(),
                y_mels.clamp(self.clamp_eps).pow(2).log10(), 
                reduction="none"
            ).mean(dim=[1,2])
        return mel_loss
    
class SISDR(nn.Module):
    """
    Scale-Invariant Source-to-Distortion Ratio
    Implementation adapted from https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py 
    """
    def __init__(self, scaling: int = True, 
                 reduction: str = "none", zero_mean: int = True):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        super().__init__()

    def forward(self, x, y):
        eps = 1e-8

        references = x.unsqueeze(1) if x.dim() == 2 else x # add channel dim
        estimates = y.unsqueeze(1) if y.dim() == 2 else y  # add channel dim

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
        sdr = 10 * torch.log10(signal/noise + eps)

        return sdr.squeeze(1) # (B,)
