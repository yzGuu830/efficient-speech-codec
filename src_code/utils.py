import collections.abc as container_abcs
import errno
import numpy as np
import os
import pickle
import torch
import torch.optim as optim
from itertools import repeat
from torchvision.utils import save_image
from config import cfg



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
    # cfg['target_size'] = dataset['train'].target_size
    return

def process_control():
    # data_shape = {'MNIST': [1, 32, 32], 'SVHN': [3, 32, 32], 'CIFAR10': [3, 32, 32], 'CIFAR100': [3, 32, 32]}
    # cfg['data_shape'] = data_shape[cfg['data_name']]
    # init_shape = {'MNIST': [7, 7], 'SVHN': [8, 8], 'CIFAR10': [8, 8], 'CIFAR100': [8, 8]}
    # cfg['init_shape'] = init_shape[cfg['data_name']]
    # cfg['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    # cfg['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    # cfg['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}
    # cfg['feature_generator'] = {'input_size': 128, 'hidden_size': 256, 'output_size': 128, 'iter': 100}
    # cfg['generator'] = {'latent_size': 1000, 'hidden_size': [128, 64], 'iter': 10}
    cfg['VQVAE'] = {}
    model_name = cfg['model_name']
    cfg[model_name]['shuffle'] = {'train': True, 'test': False}
    cfg[model_name]['optimizer_name'] = 'SGD'
    cfg[model_name]['lr'] = 1e-1
    cfg[model_name]['momentum'] = 0.9
    cfg[model_name]['weight_decay'] = 5e-4
    cfg[model_name]['nesterov'] = True
    cfg[model_name]['scheduler_name'] = 'CosineAnnealingLR'
    cfg[model_name]['num_epochs'] = 400
    cfg[model_name]['batch_size'] = {'train': 250, 'test': 500}

