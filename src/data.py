import torch, torchaudio, os, random
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
from torchaudio.functional import dcshift
from glob import glob
from tqdm import tqdm


STATS = {"mean": torch.tensor([5.24824317835737e-05, 
                              -3.034410056557135e-08]), 
         "std": torch.tensor([0.4421461522579193, 
                              0.43493565917015076])}

class LibriTTS(Dataset):
    data_name = 'LibriTTS'
    def __init__(self, data_dir, split, ft, trans_on_cpu=False) -> None:

        self.feat_trans = ft

        d_pth = '{}/{}/processed_wav/{}'.format(data_dir, self.data_name, split)

        self.source_audio = glob(f"{d_pth}/*/*.wav") # all wav paths
        self.trans_on_cpu = trans_on_cpu

    def __len__(self): # return dataset length
        return len(self.source_audio)
    
    def __getitem__(self, idx): # return ith audio waveform data 
        input = {
                'audio': torchaudio.load(self.source_audio[idx])[0][0, :-80],
                }
        # if input["audio"].mean().abs() > 1e-2:
        #     input["audio"] = dcshift(input["audio"], shift=-input["audio"].mean())

        if self.trans_on_cpu:
            input['feat'] = torch.view_as_real(self.feat_trans(input['audio'])) # [F=nfft//2-1, T=600, 2]
            # m, std = STATS["mean"], STATS["std"]
            # input['feat'] = input['feat'].sub(m).div(std)

        return input   

class DNS_CHALLENGE(Dataset):
    data_name = 'DNS_CHALLENGE'
    """https://github.com/microsoft/DNS-Challenge"""

    def __init__(self, data_dir, split, ft, trans_on_cpu=False) -> None:
        
        self.feat_trans = ft

        d_pth = '{}/DNS_CHALLENGE/processed_wav/{}'.format(data_dir, split)
        if not os.path.exists(d_pth):
            self.download_and_process(data_dir, split)

        self.source_audio = glob(f"{d_pth}/*/*.wav") # all wav paths
        self.source_audio = self.source_audio[:180000] if split == "train" else self.source_audio
        self.trans_on_cpu = trans_on_cpu

    def __len__(self): # return dataset length
        return len(self.source_audio)

    def __getitem__(self, idx): # return ith audio waveform data 
        input = {
                'audio': torchaudio.load(self.source_audio[idx])[0][0, :-80],
                }
        if self.trans_on_cpu:
            input['feat'] = torch.view_as_real(self.feat_trans(input['audio'])) # [F=nfft//2-1, T=600, 2]

        return input

    def download_and_process(self, data_dir, split):
        if os.path.exists(f"{data_dir}/DNS_CHALLENGE/processed_yz/{split}.pt"):
            file = torch.load(f"{data_dir}/DNS_CHALLENGE/processed_yz/{split}.pt")
        else:
            data_path = hf_hub_download(repo_id="Tracygu/dnscustom", 
                        filename=f"DNS_CHALLENGE/processed_yz/{split}.pt", repo_type="dataset",
                        cache_dir=data_dir, local_dir=data_dir, resume_download=True)
            file = torch.load(data_path)

        """Code to process train/test.pt files"""
        base_folder = f"{data_dir}/DNS_CHALLENGE/processed_wav/{split}"
        os.makedirs(base_folder, exist_ok=True)

        files_per_folder = 2000 if split == "train" else 1158
        num_audios = file.shape[0]
        clip_idx = 0
        for i in tqdm(range(num_audios), desc="saving audio Tensors to wav files"):
            x = file[i]
            if x.abs().sum() <= 0.001: # or x.mean().abs() >= 1e-2: # drop noisy ones
                print(f"drop {i}th clip")
            else:
                folder_num = clip_idx // files_per_folder + 1
                folder_name = f"{base_folder}/{(folder_num-1)*files_per_folder+1}-{folder_num*files_per_folder}"

                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                file_path = os.path.join(folder_name, f"clip_{clip_idx+1}.wav")

                torchaudio.save(file_path, x.unsqueeze(0), 16000)
                clip_idx += 1
        print(f"Got {clip_idx} clips for {split} set")

def fetch_dataset(data_name, data_dir, in_freq=192, win_len=20, hop_len=5, sr=16000, trans_on_cpu=False):
    dataset = {}   

    data_map = {"DNS_CHALLENGE": DNS_CHALLENGE, "LibriTTS": LibriTTS} 
    ds = data_map[data_name]
    dataset['test'] = ds(
        data_dir, split = "test",
        ft = torchaudio.transforms.Spectrogram(n_fft=(in_freq-1)*2, 
                                        win_length=int(win_len*sr*1e-3),
                                        hop_length=int(hop_len*sr*1e-3), power=None),
        trans_on_cpu = trans_on_cpu
    )
    dataset['train'] = ds(
        data_dir, split = "train",
        ft = torchaudio.transforms.Spectrogram(n_fft=(in_freq-1)*2, 
                                        win_length=int(win_len*sr*1e-3),
                                        hop_length=int(hop_len*sr*1e-3), power=None),
        trans_on_cpu = trans_on_cpu
    )
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
                     num_workers=0,
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
    
    dataset = fetch_dataset(data_name="LibriTTS", data_dir="/root/autodl-tmp/data")
    # data_loaders = make_data_loader(dataset, 
    #                  batch_size={"train": 24, "test": 16},
    #                  shuffle={"train": True, "test": False},
    #                  verbose=True)
    # for i, data in enumerate(data_loaders["test"]):
    #     print(data["audio"].shape)
    #     print(data["feat"].shape)
    #     break

    