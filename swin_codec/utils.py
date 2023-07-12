import errno
import numpy as np
import os
import pickle
import torch
import torch.optim as optim
from itertools import repeat
import collections.abc as container_abcs

import config as cfg
from pesq import pesq

def check_exists(path): # check if the path exists
    return os.path.exists(path)

def makedir_exist_ok(path): # make a directory path
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return

def save(input, path, mode='torch'): # save data
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        pickle.dump(input, open(path, 'wb'))
    else:
        raise ValueError('Not valid save mode')
    return

def load(path, mode='torch'): # load data
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return

def to_device(input, device): # 
    output = recur(lambda x, y: x.to(y), input, device)
    return output

def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    elif isinstance(input, str):
        output = input
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output

def resume(model_tag, load_tag='checkpoint', verbose=True):
    if os.path.exists('/scratch/yg172/output/model/{}_{}.pt'.format(model_tag, load_tag)):
        result = load('/scratch/yg172/output/model/{}_{}.pt'.format(model_tag, load_tag))
    else:
        print('Not exists model tag: {}, start from scratch'.format(model_tag))
        from datetime import datetime
        from logger import Logger
        last_epoch = 1
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
        result = {'epoch': last_epoch, 'logger': logger}
    if verbose:
        print('Resume from {}'.format(result['epoch']))
    return result

def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


import math
import torchaudio
from pesq import pesq
from pesq import NoUtterancesError,BufferTooShortError
import seaborn as sns
import matplotlib.pyplot as plt


def PSNR(img1, img2):
    PIXEL_MAX = max(img1.max().item(), img2.max().item())  

    img1, img2 = img1.detach().cpu().numpy(), img2.detach().cpu().numpy()

    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return 100
    
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def make_obj_score(raw, recon):
    try:
        obj_score = pesq(16000, raw, recon, 'wb')
    except NoUtterancesError as e1:
        obj_score = 0.0
    except BufferTooShortError as e2:
        obj_score = 0.0
    return obj_score

def show_and_save(raw_aud, raw_stft, recon_aud, recon_stft, path):

    psnr = PSNR(raw_stft, recon_stft)

    raw_aud, recon_aud = raw_aud.detach(), recon_aud.detach()

    obj_score = make_obj_score(raw_aud.cpu().numpy(), recon_aud.cpu().numpy())

    mel_trans = torchaudio.transforms.MelSpectrogram(n_fft=1200, win_length=320, hop_length=80, n_mels=256)
    mel_trans = mel_trans.cuda() if cfg.device == 'cuda' else mel_trans

    raw_feat, recon_feat = mel_trans(raw_aud).log(), mel_trans(recon_aud).log()

    fig, axs = plt.subplots(nrows=2, figsize=(8,3.5))

    axs[0].set_title("PESQ:{:.4f} | PSNR:{:.4f}".format(obj_score, psnr))         
    sns.heatmap(raw_feat.cpu().numpy(),ax = axs[0], cmap="mako", xticklabels=False, yticklabels=False, cbar=False).invert_yaxis()
    sns.heatmap(recon_feat.cpu().numpy(),ax = axs[1], cmap="mako", xticklabels=False, yticklabels=False, cbar=False).invert_yaxis()

    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0)

def show_and_save_multiscale(raw_aud, raw_stft, recon_auds, recon_stfts, path):

    raw_aud = raw_aud.detach()
    for i in range(len(recon_auds)):
        recon_auds[i] = recon_auds[i].detach()

    psnrs = [PSNR(raw_stft, recon_stft) for recon_stft in recon_stfts]
    obj_scores = [make_obj_score(raw_aud.cpu().numpy(), recon_aud.cpu().numpy()) for recon_aud in recon_auds]

    mel_trans = torchaudio.transforms.MelSpectrogram(n_fft=1200, win_length=320, hop_length=80, n_mels=192)
    mel_trans = mel_trans.cuda() if cfg.device == 'cuda' else mel_trans

    raw_feat, recon_feats = mel_trans(raw_aud).log(), [mel_trans(recon_aud).log() for recon_aud in recon_auds]

    nrows = len(recon_feats)+1
    fig, axs = plt.subplots(nrows=nrows, figsize=(7.5, 1.7*nrows))

    axs[0].set_title("Raw Mel Spectrogram", fontsize=8)         
    sns.heatmap(raw_feat.cpu().numpy(),ax = axs[0], cmap="mako", xticklabels=False, yticklabels=False, cbar=False).invert_yaxis()

    for i in range(1, nrows):
        axs[i].set_title("{}kbps | PESQ:{:.3f} | PSNR:{:.3f}".format(3.0*i, obj_scores[i-1], psnrs[i-1]), fontsize=8)   
        sns.heatmap(recon_feats[i-1].cpu().numpy(),ax = axs[i], cmap="mako", xticklabels=False, yticklabels=False, cbar=False).invert_yaxis()

    fig.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0)