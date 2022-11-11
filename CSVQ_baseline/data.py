import torch
import numpy as np
import models
from config import cfg, process_args
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from datasets import utils
import os


def fetch_dataset(data_name):
    from datasets.librispeech import LIBRISPEECH

    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = '/hpc/group/tarokhlab/yg172/data/{}'.format(data_name)
    if data_name in ['LIBRISPEECH']:
        dataset['train'] = eval('{}(root=root, split=\'train\')'.format(data_name))
        dataset['test'] = eval('{}(root=root, split=\'test\')'.format(data_name))
        dataset['train'].transform = utils.Compose([ 
                                                    utils.Standardize(stats=cfg['stats'][data_name])
                                                    ])                                          
        dataset['test'].transform = utils.Compose([ 
                                                    utils.Standardize(stats=cfg['stats'][data_name])
                                                    ])
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


def make_plain_transform(win_len=20, hop_len=5, sr=16e3):
    STFT = torchaudio.transforms.Spectrogram(win_length=int(win_len*sr*1e-3),hop_length=int(hop_len*sr*1e-3),power=None,return_complex=True)
    ConCat = Complex2Real()
    transf = utils.Compose([STFT, ConCat])
    return transf


class Complex2Real:
    def __call__(self, x):
        return torch.view_as_real(x)


if __name__ == '__main__':
    spklist = os.listdir('/hpc/group/tarokhlab/yg172/data/LIBRISPEECH_SMALL/raw/train-clean-100/train-clean-100/') \
    + os.listdir('/hpc/group/tarokhlab/yg172/data/LIBRISPEECH_SMALL/raw/test-clean/test-clean/')
    
    print("num_of_speakers:{}".format(len(spklist)))

    from utils import process_dataset, process_control, collate, to_device, plot_spectrogram, check_exists, makedir_exist_ok
    from models import autoencoder
    from metrics.metrics import Metric

    cfg['seed'] = 0
    dataset = fetch_dataset(cfg['data_name'])
    print(dataset)

    process_dataset(dataset)
    print(cfg)

    data_loader = make_data_loader(dataset, cfg['model_name'])
    print(data_loader)
    print(len(data_loader['train']),len(data_loader['test']))

    model = autoencoder.init_model()
    model = model.cuda()

    metric = Metric({'train': ['Loss','RECON_Loss','CODEBOOK_Loss','MEL_Loss'], 
                     'test': ['Loss','MEL_Loss','audio_PSNR','PESQ']})

    print(enumerate(data_loader['test'])[0][1].shape)
    # for i, input in enumerate(data_loader['test']):
    #     input = collate(input)
    #     input = to_device(input,'cuda')
    #     # print(input.keys())
        
    #     # print(input['stft_feat'].shape)
    #     # print(input['audio'].shape)
    #     # print(input['speaker_id'].shape)
        
    #     output = model(input)

    #     evaluation = metric.evaluate(metric.metric_name['test'], input, output)
    #     print(evaluation)

    #     if  i > 4: break
    # with torch.no_grad():
    #     root_path = "/scratch/yg172/output/runs/ohno"
    #     if not check_exists(root_path): makedir_exist_ok(root_path)
    #     img_path = os.path.join(root_path, "test_epoch_ohno.jpg")
    #     evaluation = metric.evaluate(metric.metric_name['test'], input, output)
    #     plot_spectrogram(input['stft_feat'][:4],output['recon_feat'][:4],img_path,evaluation)
        

        


    