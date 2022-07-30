import copy
import torch
import numpy as np
import models
from config import cfg, process_args
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
# from utils import collate, to_device
from datasets import utils



def fetch_dataset(data_name):
    from datasets.librispeech import LIBRISPEECH
    
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)

    if data_name in ['LIBRISPEECH']:
        dataset['train'] = eval('LIBRISPEECH(root=root, split=\'train\', '
                                'transform=utils.Compose([transforms.ToTensor()]))')
        dataset['test'] = eval('LIBRISPEECH(root=root, split=\'test\', '
                               'transform=utils.Compose([transforms.ToTensor()]))')
        dataset['train'].transform = utils.Compose([
            utils.STDCT(),
            transforms.ToTensor()
            ])
        dataset['test'].transform = utils.Compose([
            utils.STDCT(),
            transforms.ToTensor(),
            ])
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset

def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
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
    from utils import process_dataset, process_control

    process_control()
    cfg['seed'] = 0
    dataset = fetch_dataset(cfg['data_name'])
    print(dataset)

    process_dataset(dataset)
    print(cfg)

    data_loader = make_data_loader(dataset, cfg['model_name'])
    print(data_loader)