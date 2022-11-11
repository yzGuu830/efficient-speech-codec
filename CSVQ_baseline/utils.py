import errno
import numpy as np
import os
import pickle
import torch
import torch.optim as optim
from itertools import repeat
from torchvision.utils import save_image
from config import cfg
import torch.nn.functional as F


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

def process_dataset(dataset):
    cfg['data_size'] = {'train': len(dataset['train']), 'test': len(dataset['test'])}
    return

def process_control():
    data_shape = {'LIBRISPEECH':[1,128,128],'LIBRISPEECH_SMALL':[1,64,64],'CIFAR10':[3, 32, 32],'MNIST':[1, 32, 32]}
    cfg['data_shape'] = data_shape[cfg['data_name']]

    cfg['vqvae'] = {'hidden_size': 32, 'ch_mult': [1,1,2,4], 'num_res_block': 2, 'embedding_size': 512,
            'num_embedding': 16384, 'vq_commit': 0.25, 'resolution':128, 'z_channels':128, 'dropout': 0.0,
            'structure':'vqgan','ema':True,'speaker_embedding':False}

    model_name = cfg['model_name']
    cfg[model_name]['shuffle'] = {'train': True, 'test': False}
    cfg[model_name]['batch_size'] = {'train': 12, 'test': 6}
    cfg[model_name]['optimizer_name'] = 'Adam'
    cfg[model_name]['lr'] = 5.0e-06
    cfg[model_name]['momentum'] = 0.9
    cfg[model_name]['weight_decay'] = 0
    cfg[model_name]['betas'] = [0.5,0.9]
    cfg[model_name]['nesterov'] = True
    cfg[model_name]['scheduler_name'] = 'ReduceLROnPlateau'
    cfg[model_name]['factor'] = 0.85
    cfg[model_name]['patience'] = 1
    cfg[model_name]['threshold'] = 1.0e-4
    cfg[model_name]['min_lr'] = 1.0e-8
    cfg[model_name]['num_epochs'] = 50
    
def make_optimizer(model, tag):
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                              weight_decay=cfg[tag]['weight_decay'], nesterov=cfg[tag]['nesterov'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg[tag]['lr'],
                               weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer, tag):
    if cfg[tag]['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg[tag]['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
                                                   gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR': #SGD
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg[tag]['num_epochs'], eta_min=0)
    elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau': #Adam
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
                                                         patience=cfg[tag]['patience'], verbose=False,
                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
                                                         min_lr=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag]['lr'], max_lr=10 * cfg[tag]['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


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


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k])
    return input

def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse

def pad_img(x_raw,x_recon):
    if x_raw.shape[2] > x_recon.shape[2] or x_raw.shape[3] > x_recon.shape[3]:
        pad_h = x_raw.shape[2] - x_recon.shape[2] if x_raw.shape[2] - x_recon.shape[2] > 0 else 0
        pad_w = x_raw.shape[3] - x_recon.shape[3] if x_raw.shape[3] - x_recon.shape[3] > 0 else 0
        x_recon = F.pad(x_recon,(0,pad_w,0,pad_h))

    if x_raw.shape[2] < x_recon.shape[2] or x_raw.shape[3] < x_recon.shape[3]:
        pad_h = x_recon.shape[2] - x_raw.shape[2] if x_recon.shape[2] - x_raw.shape[2] > 0 else 0
        pad_w = x_recon.shape[3] - x_raw.shape[3] if x_recon.shape[3] - x_raw.shape[3] > 0 else 0
        x_raw = F.pad(x_raw,(0,pad_w,0,pad_h))
        
    return x_raw, x_recon

def pad_audio(x_raw,x_recon):
    if x_recon.shape[1] < x_raw.shape[1]:
        x_recon = F.pad(x_recon,(0,x_raw.shape[1]-x_recon.shape[1]))
    elif x_recon.shape[1] > x_raw.shape[1]:
        x_raw = F.pad(x_raw,(0,x_recon.shape[1]-x_raw.shape[1]))
    
    return x_raw, x_recon

def make_stats():
    stats = {}
    stats_path = './res/stats'
    makedir_exist_ok(stats_path)
    filenames = os.listdir(stats_path)
    for filename in filenames:
        stats_name = os.path.splitext(filename)[0]
        stats[stats_name] = load(os.path.join(stats_path, filename))
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        # data: (B,F,T, 2)
        # data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        data = data.reshape(-1,self.dim)
        # data: (B*F*T, 2)
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)

            # self.min = data.amin(dim=0)
            # self.max = data.amax(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)

            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)

            # new_min, new_max = data.amin(dim=0), data.amax(dim=0)

            old_mean, old_std  = self.mean, self.std
            # old_min, old_max = self.min, self.max

            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)

            # self.min, self.max = torch.min(old_min,new_min), torch.max(old_max,new_max)
            
            self.n_samples += n
        return

import seaborn as sns
import matplotlib.pyplot as plt
def plot_spectrogram(raw_feat, recon_feat, path, evaluation, Bs):
    '''
    raw_feat: batchsize*feats  (4*F*T*2)
    recon_feat: batchsize*feats (4*2*F*T)
    Bs: target bitstreams
    '''

    raw_feat = torch.view_as_complex(raw_feat) # 4,F,T
    recon_feat = torch.view_as_complex(recon_feat.permute(0,2,3,1).contiguous()) # 4,F,T
    
    raw_amp, recon_amp = 20*torch.log10(torch.clamp(raw_feat.abs(),1e-6)), 20*torch.log10(torch.clamp(recon_feat.abs(),1e-6))
    raw_amp, recon_amp = raw_amp.cpu(), recon_amp.cpu()

    fig, ax = plt.subplots(2, 4, figsize=(22, 10))

    evaluation = {k:round(v,4) for k,v in evaluation.items()}
    evaluation['Bitstreams'] = Bs
    fig.suptitle(str(evaluation))
    
    f_domain = raw_amp.shape[1]

    for i in range(len(ax[0])):
        ax[0][i].set_title(f"raw_feat_{i}") 
        ax[0][i].set_ylabel("Frequency(Hz):0-{}".format(f_domain))

        sns.heatmap(raw_amp[i].numpy(),ax = ax[0][i], cmap="mako", yticklabels=False).invert_yaxis()
    
    for i in range(len(ax[1])):
        ax[1][i].set_title(f"recon_feat{i}") 
        ax[1][i].set_ylabel("Frequency(Hz):0-{}".format(f_domain))
        sns.heatmap(recon_amp[i].numpy(),ax = ax[1][i], cmap="mako", yticklabels=False).invert_yaxis()

    plt.show()
    fig.savefig(path, dpi=500, bbox_inches='tight', pad_inches=0)




