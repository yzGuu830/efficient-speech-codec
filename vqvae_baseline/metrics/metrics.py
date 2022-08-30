import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils import recur
from config import cfg
from torch_dct import isdct_torch

# ViSQOL: https://github.com/google/visqol

# def PSNR(otp, target):  #>>>>>>>>#
#     otp = otp.cpu().detach().numpy()
#     target = isdct_torch(target,frame_step=256).squeeze(1).cpu().detach().numpy() 
#     print(otp.shape,target.shape)
#     if len(otp[0]) < len(target[0]):
#         otp = np.pad(otp,(0,len(target)-len(otp)))
#     elif len(otp[0]) > len(target[0]):
#         target = np.pad(target,(0,len(otp)-len(target)))
#     mse = np.mean((otp-target)**2)
#     # MAXVAL = 2**16-1
#     MAXVAL = max(np.max(otp),np.max(target))
    
#     psnr = 20*np.log10(MAXVAL/np.sqrt(mse))

#     return psnr

# PESQ: https://pypi.org/project/pesq/
'''
Sampling rate (fs|rate) - No default. Must select either 8000Hz or 16000Hz.
Note there is narrowband (nb) mode only when sampling rate is 8000Hz.
'''
def PESQ(otp, target, sr=16000, mode='wb'):  # -0.5 - 4.5
    # print(len(otp),len(target))
    from pesq import pesq
    from pesq import NoUtterancesError
    res = []
    for i in range(len(otp)):
        recon = isdct_torch(otp[i],frame_step=256).squeeze(0).cpu().detach().numpy()
        raw = target[i].squeeze(0).cpu().detach().numpy()
        # print(recon.shape,raw.shape)
        try:
            sc = pesq(sr, recon, raw, mode)
            res.append(sc)
        except NoUtterancesError as e:
            pass
        
    avg_score = np.mean(np.array(res))
    return avg_score

class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'Perplexity': (lambda input, output: output['perplexity'].item()),
                       'PSNR': (lambda input, output: PSNR(output['x_prime'], input['raw'])),
                       'PESQ': (lambda input, output: PESQ(output['x_prime'], input['raw']))   # pesq ranges from -0.5 to 4.5
                       }

    def make_metric_name(self, metric_name):
        return metric_name

    def make_pivot(self):
        if cfg['data_name'] in ['LIBRISPEECH']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'PESQ'
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