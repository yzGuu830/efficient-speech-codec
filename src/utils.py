import os
import torch
import pickle
import numpy as np
import errno

from pesq import pesq, NoUtterancesError, BufferTooShortError
import librosa
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
import wandb, math


def PSNR(img1, img2):
    PIXEL_MAX = max(img1.max().item(), img2.max().item()) 
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def PESQ(raw, recon):
    try:
        obj_score = pesq(16000, raw, recon, 'wb')
    except NoUtterancesError as e1:
        obj_score = 0.0
    except BufferTooShortError as e2:
        obj_score = 0.0
    return obj_score

def show_and_save(raw_aud, recon_aud, save_path=None, use_wb=False):
    x1 = raw_aud.squeeze().cpu().numpy()
    x2 = recon_aud.squeeze().cpu().numpy()

    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(x1, n_fft=1024, win_length=320, hop_length=80)), ref=np.max)
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(x2, n_fft=1024, win_length=320, hop_length=80)), ref=np.max)
    score = round(PESQ(x1, x2), 2)

    if use_wb:
        wandb.log({"PESQ_val": score})
        val_img_log = {"raw": wandb.Image(np.flip(D1, axis=0)), "recon":wandb.Image(np.flip(D2, axis=0))}
        wandb.log(val_img_log)
    else:
        nrows = 2
        fig, ax = plt.subplots(figsize=(7.5,1.7*nrows), nrows=nrows, ncols=1, sharex=True)
        img = librosa.display.specshow(D1, y_axis='linear', x_axis='time', sr=16000, ax=ax[0])
        ax[0].set(title='Raw Audio')
        ax[0].label_outer()
        img_ = librosa.display.specshow(D2, y_axis='linear', x_axis='time', sr=16000, ax=ax[1])
        ax[1].set(title=f'Recon Audio PESQ:{score}')
        ax[1].label_outer()
        fig.colorbar(img, ax=ax, format="%+2.f dB")

        if save_path:
            fig.savefig(f"{save_path}/spec.jpg", dpi=150, bbox_inches='tight', pad_inches=0)
            torchaudio.save(f"{save_path}/raw.wav", raw_aud.cpu(), 16000)
            torchaudio.save(f"{save_path}/recon.wav", recon_aud.cpu(), 16000)


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