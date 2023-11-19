import torch, torchaudio
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from torch.utils.data import Dataset

from utils import load

seed = 53
win_len, hop_len = 20, 5
sr = 16000

class DNS_CHALLENGE(Dataset):
    data_name = 'DNS_CHALLENGE'
    """https://github.com/microsoft/DNS-Challenge"""

    def __init__(self, data_dir, split, ft) -> None:
        
        self.split = split

        self.feat_trans = ft

        self.source_audio = load('{}/{}.pt'.format(data_dir, self.split), mode='torch')

    def __len__(self): # return dataset length
        return self.source_audio.size(0)

    def __getitem__(self, idx): # return ith audio waveform data 
        input = {'audio':self.source_audio[idx][:-80]}
        input['feat'] = torch.view_as_real(self.feat_trans(input['audio'])) # [F=nfft//2-1, T=600, 2]

        return input

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str
    

def fetch_dataset(data_name, data_dir, in_freq=192):
    dataset = {}    
    if data_name in ['DNS_CHALLENGE']:
        dataset['train'] = DNS_CHALLENGE(
            data_dir, split = "train",
            ft = torchaudio.transforms.Spectrogram(n_fft=(in_freq-1)*2, 
                                            win_length=int(win_len*sr*1e-3),
                                            hop_length=int(hop_len*sr*1e-3), power=None)
        )
        dataset['test'] = DNS_CHALLENGE(
            data_dir, split = "test",
            ft = torchaudio.transforms.Spectrogram(n_fft=(in_freq-1)*2, 
                                            win_length=int(win_len*sr*1e-3),
                                            hop_length=int(hop_len*sr*1e-3), power=None)
        )
    else:
        raise ValueError('Not valid dataset name')
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

def make_data_loader(dataset, 
                     batch_size, 
                     shuffle, 
                     verbose=True,
                     sampler=None,
                     num_workers=1,):
    data_loader = {}
    for k in dataset:
        _batch_size, _shuffle = batch_size[k], shuffle[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(seed))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=False,
                                        sampler=sampler[k], num_workers=num_workers,
                                        pin_memory=True, collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(seed))
    if verbose: 
        print("data ready")
    return data_loader

if __name__ == "__main__":
    
    dataset = fetch_dataset(data_name="DNS_CHALLENGE")
    data_loaders = make_data_loader(dataset, 
                     batch_size={"train": 24, "test": 16},
                     shuffle={"train": True, "test": False},
                     verbose=True)
    for i, data in enumerate(data_loaders["test"]):
        print(data["audio"].shape)
        print(data["feat"].shape)
        break