import numpy as np
import sys
import os
import torch
import torch.nn.functional as F
from config import cfg
sys.path.append(os.path.abspath(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from pesq import pesq
from pesq import NoUtterancesError,BufferTooShortError

def audio_MSE(recon_audio, audio):
    with torch.no_grad():
        return F.mse_loss(raw,recon,reduction='mean').item()
        

def audio_MAE(recon_audio,audio):
    with torch.no_grad():
        return F.l1_loss(raw,recon,reduction='mean').item()

def audio_PSNR(recon_audio, audio):
    with torch.no_grad():
        max_I = torch.max(torch.max(recon_audio), torch.max(audio))
        return 20*(torch.log10(max_I/(torch.sqrt(F.mse_loss(recon_audio, audio, reduction='mean'))))).item()            

def img_MSE(recon, raw):
    with torch.no_grad(): 
        raw, recon = raw.type(torch.float64), recon.type(torch.float64)
        return F.mse_loss(raw,recon,reduction='mean').item()

def img_PSNR(recon, raw):  # otp: batch of 2-D features  target: batch of 1-D raw audios
    with torch.no_grad(): 
        raw, recon = raw.type(torch.float64), recon.type(torch.float64)
        mse = F.mse_loss(raw,recon,reduction='mean')
        max_I = torch.max(torch.amax(raw),torch.amax(recon))
        return 20*(torch.log10(max_I / (torch.sqrt(mse)))).item()

# PESQ: https://pypi.org/project/pesq/
'''
Sampling rate (fs|rate) - No default. Must select either 8000Hz or 16000Hz.
Note there is narrowband (nb) mode only when sampling rate is 8000Hz.
'''
def PESQ(recon_audio, audio):  # -0.5 - 4.5
    with torch.no_grad():
        score, invalid = 0, 0
        for i in range(audio.size(0)):
            raw, recon = audio[i].cpu().numpy(),recon_audio[i].cpu().numpy()
            try:
                sc = pesq(16000, raw, recon, 'wb')
                score += sc
            except NoUtterancesError as e1:
                invalid += 1
            except BufferTooShortError as e2:
                invalid += 1
    if audio.size(0) == invalid: return -0.5
    if invalid / audio.size(0) > 0.5: print("Warning: too much invalid reconstructions")
    return score / (audio.size(0) - invalid)            

class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {
                       'Loss': (lambda input, output: output['loss'].mean().item() if cfg['num_workers'] > 0 else output['loss'].item()),
                    #    'Perplexity': (lambda input, output: output['ppl'].item()),
                       'audio_MSE':  (lambda input, output: audio_MSE(output['recon_audio'], input['audio'])),
                       'audio_MAE':  (lambda input, output: audio_MAE(output['recon_audio'], input['audio'])),
                       'audio_PSNR': (lambda input, output: audio_PSNR(output['recon_audio'], input['audio'])),
                       'img_MSE': (lambda input, output: img_MSE(output['recon_feat'], input['stft_feature'])),
                       'img_PSNR': (lambda input, output: img_PSNR(output['recon_feat'], input['stft_feature'])),
                       'PESQ': (lambda input, output: PESQ(output['recon_audio'], input['audio'])),   # pesq ranges from -0.5 to 4.5
                       'RECON_Loss': (lambda input, output: output['recon_loss'].mean().item() if cfg['num_workers'] > 0 else output['recon_loss'].item()),
                       'CODEBOOK_Loss': (lambda input, output: output['vq_loss'].mean().item() if cfg['num_workers'] > 0 else output['vq_loss'].item()),
                       'MEL_Loss': (lambda input, output: output['mel_loss'].mean().item() if cfg['num_workers'] > 0 else output['mel_loss'].item()),
                       }
        
    def make_metric_name(self, metric_name):
        return metric_name

    def make_pivot(self):
        if cfg['data_name'] in ['LIBRISPEECH','LIBRISPEECH_SMALL', 'DNS_CHALLENGE']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'PESQ'
        elif cfg['data_name'] in ['CIFAR10','MNIST']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'img_PSNR'
        else:
            raise ValueError('Not valid data name')
        return pivot, pivot_name, pivot_direction

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return

# if __name__ == "__main__":
    # metric = Metric({'train': ['Loss', 'PSNR', 'PESQ'], 'test': ['Loss', 'PSNR', 'PESQ']})