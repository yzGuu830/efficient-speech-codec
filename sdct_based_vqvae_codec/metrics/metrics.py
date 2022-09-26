import numpy as np
import sys
import os
import torch
import torch.nn.functional as F
from utils import recur, pad_audio
from config import cfg
sys.path.append(os.path.abspath(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# ViSQOL: https://github.com/google/visqol

def audio_MSE(recon_audio, audio):
    with torch.no_grad():
        avg_mse = 0
        for i in range(len(audio)):
            raw, recon = pad_audio(audio[i],recon_audio[i])
            mse = F.mse_loss(raw,recon,reduction='mean').item()
            avg_mse += mse
    return avg_mse/len(audio)

def audio_PSNR(recon_audio, audio):  # otp: batch of 2-D features  target: batch of 1-D raw audios
    with torch.no_grad():
        avg_psnr = 0
        for i in range(len(audio)):
            raw, recon = pad_audio(audio[i],recon_audio[i])
            mse = F.mse_loss(raw,recon,reduction='mean')
            max_I = torch.max(torch.max(raw),torch.max(recon))
            psnr = 20*(torch.log10(max_I / (torch.sqrt(mse)) )).item()
            avg_psnr += psnr
    return avg_psnr/len(audio)

def img_MSE(recon, raw):
    with torch.no_grad(): 
        return F.mse_loss(raw,recon,reduction='mean').item()

def img_PSNR(recon, raw):  # otp: batch of 2-D features  target: batch of 1-D raw audios
    with torch.no_grad(): 
        mse = F.mse_loss(raw,recon,reduction='mean')
        max_I = torch.max(torch.max(raw),torch.max(recon))
        return 20*(torch.log10(max_I / (torch.sqrt(mse)))).item()  


# PESQ: https://pypi.org/project/pesq/
'''
Sampling rate (fs|rate) - No default. Must select either 8000Hz or 16000Hz.
Note there is narrowband (nb) mode only when sampling rate is 8000Hz.
'''
def PESQ(recon_audio, audio):  # -0.5 - 4.5
    from pesq import pesq
    from pesq import NoUtterancesError
    with torch.no_grad():
        avg_pesq, invalid = 0, 0
        for i in range(len(audio)):
            raw, recon = pad_audio(audio[i],recon_audio[i])
            try:
                sc = pesq(16000, recon.squeeze().cpu().numpy(), raw.squeeze().cpu().numpy(), 'wb')
                avg_pesq += sc
            except NoUtterancesError as e:
                invalid += 1
    if len(audio)-invalid == 0: return -0.5
    return avg_pesq/(len(audio)-invalid)

class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'audio_MSE':  (lambda input, output: audio_MSE(output['recon_audio'], input['audio'])),
                       'audio_PSNR': (lambda input, output: audio_PSNR(output['recon_audio'], input['audio'])),
                       'img_MSE': (lambda input, output: img_MSE(output['recon_feature'], input['feature'])),
                       'img_PSNR': (lambda input, output: img_PSNR(output['recon_feature'], input['feature'])),
                       'PESQ': (lambda input, output: PESQ(output['recon_audio'], input['audio']))   # pesq ranges from -0.5 to 4.5
                       }
        # self.metric = {'Loss': (lambda input, output: output['loss'].item()),
        #                'img_MSE':  (lambda input, output: img_MSE(output['recon_feature'], input['data'])),
        #                'img_PSNR': (lambda input, output: img_PSNR(output['recon_feature'], input['data'])),
        #                }
    def make_metric_name(self, metric_name):
        return metric_name

    def make_pivot(self):
        if cfg['data_name'] in ['LIBRISPEECH']:
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