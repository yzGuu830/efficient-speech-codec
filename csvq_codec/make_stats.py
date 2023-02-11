import os
import torch
import datasets
from config import cfg
from data import fetch_dataset, make_data_loader
from utils import save, collate, Stats, makedir_exist_ok, process_control, process_dataset

from tqdm import tqdm

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    stats_path = '/hpc/group/tarokhlab/yg172/res/stats'
    dim = 2
    data_names = ['DNS_CHALLENGE']
    # process_control()
    cfg['seed'] = 0
    with torch.no_grad():
        for data_name in data_names:
            dataset = fetch_dataset(data_name)
            # dataset['train'].transform=datasets.Compose([transforms.ToTensor()])
            process_dataset(dataset)
            data_loader = make_data_loader(dataset, 'csvq_codec')
            stats = Stats(dim=dim)
            for i, input in tqdm(enumerate(data_loader['train'])):
                # input = collate(input)
                stats.update(input['stft_feat'])
            stats = (stats.mean.tolist(), stats.std.tolist())
            print(data_name, stats)
            makedir_exist_ok(stats_path)
            save(stats, os.path.join(stats_path, '{}.pt'.format(data_name)))