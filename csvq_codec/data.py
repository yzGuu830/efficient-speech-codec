import torch
import numpy as np
from config import cfg, process_args
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from dataset import utils
from dataset.librispeech import LIBRISPEECH
from dataset.dnsv5 import DNSV5



def fetch_dataset(data_name):

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
    elif data_name in ['DNS_CHALLENGE']:
        dataset['train'] = eval('{}(root=root, split=\'train\')'.format('DNSV5'))
        dataset['test'] = eval('{}(root=root, split=\'test\')'.format('DNSV5'))
        # dataset['train'].transform = utils.Compose([ 
        #                                             utils.Standardize(stats=cfg['stats'][data_name])
        #                                             ])                                          
        # dataset['test'].transform = utils.Compose([ 
        #                                             utils.Standardize(stats=cfg['stats'][data_name])
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

        for key in output:
            output[key] = torch.stack(output[key],dim=0)

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
    
    # from utils import process_dataset
    # from models import autoencoder
    # from metrics.metrics import Metric

    cfg['data_name'] = 'LIBRISPEECH'

    cfg['seed'] = 0
    dataset = fetch_dataset(cfg['data_name'])
    # process_dataset(dataset)

    data_loader = make_data_loader(dataset, cfg['model_name'])
    print(data_loader)
    print(len(data_loader['train']),len(data_loader['test']))


    # model = autoencoder.init_model()
    # model = model.cuda()

    # metric = Metric({'train': ['Loss','RECON_Loss','CODEBOOK_Loss','MEL_Loss'], 
    #                  'test': ['Loss','MEL_Loss','audio_PSNR','PESQ']})

    for i, input in enumerate(data_loader['train']):
        print(input['stft_feat'].shape)
        break
    
    for i, input in enumerate(data_loader['test']):
        print(input['stft_feat'].shape)
        break
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
        

        


    