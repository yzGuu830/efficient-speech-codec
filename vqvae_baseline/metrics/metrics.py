import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils import recur
from config import cfg

# ViSQOL: https://github.com/google/visqol

def PSNR(output, target):
    # print(len(output),len(target))
    if len(output) < len(target):
        output = np.pad(output,(0,len(target)-len(output)))
    mse = np.mean((output-target)**2)
    # MAXVAL = 2**16-1
    MAXVAL = max(max(output),max(target))
    
    psnr = 20*np.log10(MAXVAL/np.sqrt(mse))

    return psnr

# PESQ: https://pypi.org/project/pesq/
def PESQ(output, target, sr, mode='wb'):
    from pesq import pesq
    return pesq(sr, output, target,mode)


class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'PSNR': (lambda input, output: recur(PSNR, output['recon_audio'], input['data'])),
                       'PESQ': (lambda input, output: recur(PESQ, output['recon_audio'], input['data']))   # pesq ranges from -0.5 to 4.5
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