from torch.utils.data import Dataset
import torchaudio
import glob
import os
import numpy as np
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file #, make_classes_counts, make_tree, make_flat_index
import torch

class LIBRISPEECH(Dataset):
    data_name = 'LIBRISPEECH'
    file = [('http://www.openslr.org/resources/12/dev-clean.tar.gz','42e2234ba48799c1f50f24a7926300a1'),
            ('http://www.openslr.org/resources/12/test-clean.tar.gz','32fa31d27d2e1cad72775fee3f4849a9'),
            ('http://www.openslr.org/resources/12/train-clean-100.tar.gz','2a93770f6d5c6c964bc36631d331a522')]

    def __init__(self, root, split, transform=None) -> None:
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()

        id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)),
                                          mode='pickle')
        self.other = {'id': id}

    def __len__(self): # return dataset length
        return len(self.data)

    def __getitem__(self, idx): # return ith audio waveform data 
        x_raw, x_reconst = self.data[idx], self.target[idx]
        input = {'data': x_raw, 'target': x_reconst}
        # using stdct here
        if self.transform is not None:
            input = self.transform(input)
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

        train_set, test_set, dev_set  = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(dev_set, os.path.join(self.processed_folder, 'dev.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            # print(filename)
            # if not check_exists(f'./data/LIBRISPEECH/raw/{filename}'):
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
            os.rename(f'./data/{self.data_name}/raw/LibriSpeech', f'./data/{self.data_name}/raw/{filename.split(".")[0]}')
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        # glob.glob("LibriSpeech/LibriSpeech_dev-clean/dev-clean"+"/*/*/*.flac")
        dev_audio_path = f"{self.root}/raw/dev-clean/*/*/*/*.flac"
        dev_audio_files = glob.glob(dev_audio_path)
        

        test_audio_path = f"{self.root}/raw/test-clean/*/*/*/*.flac"
        test_audio_files = glob.glob(test_audio_path)

        train_audio_path = f"{self.root}/raw/train-clean-100/*/*/*/*.flac"
        train_audio_files = glob.glob(train_audio_path)

        def audio_load(files):
            data = []
            for f in files:
                wf, _ = torchaudio.load(f)
                data.append(wf.numpy())
            return data

        dev_data, test_data, train_data = audio_load(dev_audio_files), audio_load(test_audio_files), audio_load(train_audio_files)
        dev_id, test_id, train_id = np.arange(len(dev_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64), np.arange(len(train_data)).astype(np.int64)
        
        return (train_id, train_data, train_data), (test_id, test_data, test_data), (dev_id, dev_data, dev_data)
