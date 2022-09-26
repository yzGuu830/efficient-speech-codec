from torch.utils.data import Dataset
import torchaudio
import glob
import os
import numpy as np
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, STDCT, get_sign
import torch
from torch_dct import sdct_torch, isdct_torch


class LIBRISPEECH(Dataset):
    data_name = 'LIBRISPEECH'
    file = [ #('http://www.openslr.org/resources/12/dev-clean.tar.gz','42e2234ba48799c1f50f24a7926300a1'),
            ('http://www.openslr.org/resources/12/test-clean.tar.gz','32fa31d27d2e1cad72775fee3f4849a9'),
            ('http://www.openslr.org/resources/12/train-clean-100.tar.gz','2a93770f6d5c6c964bc36631d331a522')]

    def __init__(self, root, split, transform=None) -> None:
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.dataset = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)), mode='torch')

    def __len__(self): # return dataset length
        if isinstance(self.dataset,tuple):
            return len(self.dataset[0])
        else:
            return len(self.dataset)

    def __getitem__(self, idx): # return ith audio waveform data 
        input = {'audio':self.dataset[0][idx], 'sdct_feat':self.dataset[1][idx], 'sign':self.dataset[2][idx]}
        if self.transform is not None: # Standardize
            output = self.transform(input['sdct_feat'])
            for key in output.keys():
                input[key] = output[key]
            input['feature'] = input.pop('power_feat')
        return input

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        else: print("data downloaded")
        train_data, train_feature, train_sign, test_data, test_feature, test_sign = self.make_data()

        save((train_data,train_feature,train_sign), os.path.join(self.processed_folder, 'train.pt'), mode='torch')
        save((test_data,test_feature,test_sign), os.path.join(self.processed_folder, 'test.pt'), mode='torch')
        
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
            os.rename(f'{self.root}/raw/LibriSpeech', f'{self.root}/raw/{filename.split(".")[0]}')
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        # glob.glob("LibriSpeech/LibriSpeech_dev-clean/dev-clean"+"/*/*/*.flac")
        # dev_audio_path = f"{self.root}/raw/dev-clean/*/*/*/*.flac"
        # dev_audio_files = glob.glob(dev_audio_path)
        
        test_audio_path = f"{self.root}/raw/test-clean/*/*/*/*.flac"
        test_audio_files = glob.glob(test_audio_path)

        train_audio_path = f"{self.root}/raw/train-clean-100/*/*/*/*.flac"
        train_audio_files = glob.glob(train_audio_path)

        def batchify(audio_files,length=8000):
            c = torch.Tensor()
            data = []
            count = 0
            for f in audio_files:
                wf, _ = torchaudio.load(f)
                c = torch.cat((c,wf),dim=1)
                while c.shape[1]>length:
                    if STDCT()(c[:,:length]).std().item() == 0:
                        count+=1
                        print('Useless Feature count: {} Useful Feature count: {}'.format(count,len(data)))
                    else:
                        data.append(c[:,:length])
                    c = c[:,length:]
            data = torch.stack(data)
            return data
        
        def get_feature(audios):
            feat, sign = [], []
            for audio in audios:
                sdct_feat = STDCT()(audio)
                feat.append(sdct_feat)
                sign.append(get_sign(sdct_feat))
            return torch.stack(feat), torch.stack(sign)

        test_data = batchify(test_audio_files,length=80000)
        train_data = batchify(train_audio_files,length=8000)
        print("Batchify Finish")
        train_feature, train_sign = get_feature(train_data)
        test_feature, test_sign = get_feature(test_data)
        print("Feature Computing Finish")

        return train_data, train_feature, train_sign, test_data, test_feature, test_sign

