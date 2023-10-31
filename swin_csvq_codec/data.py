import torch
import numpy as np
import config as cfg
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

# from dataset.librispeech import LIBRISPEECH
from dataset.dnsv5 import DNSV5

def fetch_dataset(data_name):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = '{}/{}'.format(cfg.data_path, data_name)
    if data_name in ['LIBRISPEECH']:
        dataset['train'] = eval('{}(root=root, split=\'train\')'.format(data_name))
        dataset['test'] = eval('{}(root=root, split=\'test\')'.format(data_name))

    elif data_name in ['DNS_CHALLENGE']:
        dataset['train'] = eval('{}(root=root, split=\'train\')'.format('DNSV5'))
        dataset['test'] = eval('{}(root=root, split=\'test\')'.format('DNSV5'))

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

        for key in output:
            output[key] = torch.stack(output[key],dim=0)

        return output
    else:
        return default_collate(batch)

def make_data_loader(dataset, batch_size, shuffle, verbose=True, sampler=None):
    data_loader = {}
    if verbose: print(f"Batch_size: Train {batch_size['train']} | Test {batch_size['test']}")
    for k in dataset:
        _batch_size, _shuffle = batch_size[k], shuffle[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, num_workers=cfg.num_workers, collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg.seed))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=True, num_workers=cfg.num_workers, collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg.seed))
    return data_loader

if __name__ == "__main__":
    import collections
    print(cfg)
    cfg = collections.OrderedDict(cfg)

    print(cfg)