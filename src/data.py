import torch, torchaudio, os
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from torch.utils.data import Dataset
from utils import load
from huggingface_hub import hf_hub_download
from glob import glob
from tqdm import tqdm


class DNS_CHALLENGE(Dataset):
    data_name = 'DNS_CHALLENGE'
    """https://github.com/microsoft/DNS-Challenge"""

    def __init__(self, data_dir, split, ft, trans_on_cpu=False) -> None:
        
        self.feat_trans = ft

        d_pth = '{}/DNS_CHALLENGE/processed_wav/{}'.format(data_dir, split)
        if not os.path.exists(d_pth):
            self.download_and_process(data_dir, split)

        self.source_audio = glob(f"{d_pth}/*.wav") # all wav paths
        self.trans_on_cpu = trans_on_cpu

    def __len__(self): # return dataset length
        return len(self.source_audio)

    def __getitem__(self, idx): # return ith audio waveform data 
        input = {
                'audio': torchaudio.load(self.source_audio[idx])[0][0, :-80]
                }
        if self.trans_on_cpu:
            input['feat'] = torch.view_as_real(self.feat_trans(input['audio'])) # [F=nfft//2-1, T=600, 2]
        # input['feat'] = None
        return input

    def download_and_process(self, data_dir, split):
        data_path = hf_hub_download(repo_id="Tracygu/dnscustom", 
                    filename=f"DNS_CHALLENGE/processed_yz/{split}.pt", repo_type="dataset",
                    cache_dir=data_dir, local_dir=data_dir, resume_download=True)
        """Code to process train/test.pt files"""
        file = torch.load(data_path)
        os.makedirs(f"{data_dir}/DNS_CHALLENGE/processed_wav/{split}", exist_ok=True)
        for i in tqdm(range(file.size(0)), desc="saving audio Tensors to wav files"):
            torchaudio.save(f"{data_dir}/DNS_CHALLENGE/processed_wav/{split}/clip_{i}.wav", src=file[i:i+1], sample_rate=16000,)

def fetch_dataset(data_name, data_dir, in_freq=192, win_len=20, hop_len=5, sr=16000, trans_on_cpu=False):
    dataset = {}    
    if data_name in ['DNS_CHALLENGE']:
        dataset['test'] = DNS_CHALLENGE(
            data_dir, split = "test",
            ft = torchaudio.transforms.Spectrogram(n_fft=(in_freq-1)*2, 
                                            win_length=int(win_len*sr*1e-3),
                                            hop_length=int(hop_len*sr*1e-3), power=None),
            trans_on_cpu = trans_on_cpu
        )
        dataset['train'] = DNS_CHALLENGE(
            data_dir, split = "train",
            ft = torchaudio.transforms.Spectrogram(n_fft=(in_freq-1)*2, 
                                            win_length=int(win_len*sr*1e-3),
                                            hop_length=int(hop_len*sr*1e-3), power=None),
            trans_on_cpu = trans_on_cpu
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

        output['feat'] = None
        return output
    else:
        return default_collate(batch)

def make_data_loader(dataset, 
                     batch_size, 
                     shuffle, 
                     verbose=True,
                     sampler=None,
                     num_workers=1,
                     seed=0,):
    data_loader = {}
    for k in dataset:
        _batch_size, _shuffle = batch_size[k], shuffle[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, collate_fn=input_collate, num_workers=num_workers,
                                        worker_init_fn=np.random.seed(seed)
                                        )
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=False,
                                        sampler=sampler[k], num_workers=num_workers,
                                        pin_memory=False, collate_fn=input_collate,
                                        # worker_init_fn=np.random.seed(seed)
                                        )
    if verbose: 
        print("data ready")
    return data_loader

if __name__ == "__main__":
    
    dataset = fetch_dataset(data_name="DNS_CHALLENGE")
    # data_loaders = make_data_loader(dataset, 
    #                  batch_size={"train": 24, "test": 16},
    #                  shuffle={"train": True, "test": False},
    #                  verbose=True)
    # for i, data in enumerate(data_loaders["test"]):
    #     print(data["audio"].shape)
    #     print(data["feat"].shape)
    #     break