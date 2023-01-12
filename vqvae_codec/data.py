import copy
import torch
import numpy as np
import models
from config import cfg, process_args
from torchvision import transforms
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
# from utils import collate, to_device
from datasets import utils

data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))}

def fetch_dataset(data_name):
    from datasets.librispeech import LIBRISPEECH
    from datasets.cifar import CIFAR10, CIFAR100
    from datasets.mnist import MNIST
    
    dataset = {}
    print('fetching data {}...'.format(data_name))
    # root = './data/{}'.format(data_name)
    root = '/hpc/group/tarokhlab/yg172/data/{}'.format(data_name)

    if data_name in ['LIBRISPEECH'] and cfg['model_name'] in ['vqvae']:
        dataset['train'] = eval('LIBRISPEECH(root=root, split=\'train\')')
        dataset['test'] = eval('LIBRISPEECH(root=root, split=\'test\')')
        dataset['train'].transform = utils.Compose([utils.Standardize(),
                                                    utils.PowerSpectrogram(),
                                                    utils.Standardize()
                                                    ])                                          
        dataset['test'].transform = utils.Compose([ utils.Standardize(),
                                                    utils.PowerSpectrogram(),
                                                    utils.Standardize()
                                                    ])
    elif data_name in ['LIBRISPEECH'] and cfg['model_name'] in ['vqvae_wavenet']:
        dataset['train'] = eval('LIBRISPEECH(root=root, split=\'train\')')
        dataset['test'] = eval('LIBRISPEECH(root=root, split=\'test\')')
        sr = 16e3
        cfg['data_length'] = 0.5 * sr
        cfg['n_fft'] = round(0.02 * sr)
        cfg['hop_length'] = round(0.01 * sr)
        # cfg['background_noise'] = dataset['train'].background_noise
        train_transform = make_transform('plain')
        test_transform = make_transform('plain')
        dataset['train'].transform = datasets.Compose(
            [train_transform, torchvision.transforms.Normalize(*cfg['stats'][data_name])])
        dataset['test'].transform = datasets.Compose(
            [test_transform, torchvision.transforms.Normalize(*cfg['stats'][data_name])])


    elif data_name in ['MNIST']:
        dataset['train'] = eval('MNIST(root=root, split=\'train\')')
        dataset['test'] = eval('MNIST(root=root, split=\'test\')')

        # dataset['train'].transform = utils.Compose([transforms.PILToTensor(),
        #                                             transforms.ConvertImageDtype(torch.float),
        #                                             transforms.Normalize(*data_stats[data_name])])
        # dataset['test'].transform  = utils.Compose([transforms.PILToTensor(),
        #                                             transforms.ConvertImageDtype(torch.float),
        #                                             transforms.Normalize(*data_stats[data_name])])
        dataset['train'].transform = utils.Compose([transforms.PILToTensor(),
                                                    transforms.ConvertImageDtype(torch.float)
                                                    ])
        dataset['test'].transform  = utils.Compose([transforms.PILToTensor(),
                                                    transforms.ConvertImageDtype(torch.float)
                                                    ])
    elif data_name in ['CIFAR10']:
        dataset['train'] = eval('CIFAR10(root=root, split=\'train\')')
        dataset['test'] = eval('CIFAR10(root=root, split=\'test\')')

        dataset['train'].transform = utils.Compose([transforms.PILToTensor(),
                                                    transforms.ConvertImageDtype(torch.float),
                                                    transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform  = utils.Compose([transforms.PILToTensor(),
                                                    transforms.ConvertImageDtype(torch.float),
                                                    transforms.Normalize(*data_stats[data_name])])

        # dataset['train'].transform = utils.Compose([transforms.PILToTensor(),
        #                                             transforms.ConvertImageDtype(torch.float)
        #                                             ])
        # dataset['test'].transform  = utils.Compose([transforms.PILToTensor(),
        #                                             transforms.ConvertImageDtype(torch.float)
        #                                             ])
    
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset

def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}  # output = {'audio':[.....], 'feature':[.....]}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        # pad to same length
        # if 'feature' in output.keys():
        #     padded_feat = torch.nn.utils.rnn.pad_sequence([feat.transpose(1,2).squeeze(0) for feat in output['feature']],batch_first=True)
        #     padded_feat = padded_feat.unsqueeze_(1)
        #     output['feature'] = padded_feat #(N=batchsize,C=1,H,W)
        return output
    else:
        return default_collate(batch)

def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
    return data_loader


if __name__ == '__main__':
    from utils import process_dataset, process_control, collate

    process_control()
    cfg['seed'] = 0
    dataset = fetch_dataset(cfg['data_name'])
    print(dataset)

    process_dataset(dataset)
    print(cfg)

    data_loader = make_data_loader(dataset, cfg['model_name'])
    print(data_loader)
    print(len(data_loader['train']),len(data_loader['test']))
    for i, input in enumerate(data_loader['train']):
        print(input.keys())
        print(len(input['sign']))
        print(len(input['sdct_stats']))
        input = collate(input)

        print(input['sign'][0])

        print(input['sdct_stats'][0])
        break


    