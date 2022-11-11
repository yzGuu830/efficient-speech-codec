from torch.utils.data import Dataset
import torchaudio
import torch
import glob
import os
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file
from config import cfg
from data import make_plain_transform

class LIBRISPEECH(Dataset):
    data_name = 'LIBRISPEECH'
    file = [('http://www.openslr.org/resources/12/test-clean.tar.gz','32fa31d27d2e1cad72775fee3f4849a9'),
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
        input = {'audio':self.dataset[0][idx], 'stft_feat':self.dataset[1][idx], 'speaker_id':self.dataset[2][idx]}
        if self.transform is not None:
            input['stft_feat'] = self.transform(input['stft_feat'])
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
        train_data, train_feature, train_spk_ids, test_data, test_feature, test_spk_ids = self.make_data()

        save((train_data,train_feature,train_spk_ids), os.path.join(self.processed_folder, 'train.pt'), mode='torch')
        save((test_data,test_feature,test_spk_ids), os.path.join(self.processed_folder, 'test.pt'), mode='torch')
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
        test_audio_path = f"{self.root}/raw/test-clean/*/*/*/*.flac"
        test_audio_files = glob.glob(test_audio_path)     # 2620

        train_audio_path = f"{self.root}/raw/train-clean-100/*/*/*/*.flac"
        train_audio_files = glob.glob(train_audio_path)   # 28539

        speaker_id = os.listdir(f"{self.root}/raw/train-clean-100/train-clean-100/") + os.listdir(f"{self.root}/raw/test-clean/test-clean/")
        speaker_id_map = {}
        for i in range(len(speaker_id)):
            # speaker id, speaker idx
            speaker_id_map[speaker_id[i]] = i
        
        transf = make_plain_transform(win_len=cfg['win_length'],hop_len=cfg['hop_length'],sr=cfg['sr'])

        def batchify(audio_files, length, num_clips):
            print(f"Audio Length: {length} Number Clips: {num_clips}")
            c = torch.Tensor()
            data, feat, ids = [], [], []
            count = 0
            for f in audio_files:
                mapped_id = speaker_id_map[(f.split('/')[-1]).split('-')[0]]
                wf, _ = torchaudio.load(f)
                c = torch.cat((c,wf),dim=1)

                if len(data) > num_clips: 
                    print(f"GOT {num_clips} Features")
                    break

                while c.shape[1] > length:
                    audio = c[:,:length]
                    stft_feat = transf(audio)
                    if stft_feat.max().item() < 1e-5 or stft_feat.std().item() == 0:
                        count+=1
                        print('Useless Feature count: {} Useful Feature count: {}'.format(count,len(data)))
                    else:
                        data.append(audio)
                        feat.append(stft_feat)
                        ids.append(mapped_id)
                    c = c[:,length:]

                c = torch.Tensor()
            return torch.cat(data,dim=0), torch.cat(feat,dim=0), torch.tensor(ids).unsqueeze(1)

        train_data, train_feature, train_spk_ids = batchify(train_audio_files, length=int((cfg['train_dur']-cfg['hop_length']*0.001)*cfg['sr']), num_clips=cfg['train_clips'])
        print("Train Batchify Finish")
        test_data, test_feature, test_spk_ids = batchify(test_audio_files,length=int((cfg['test_dur']-cfg['hop_length']*0.001)*cfg['sr']), num_clips=cfg['test_clips'])
        print("Test Batchify Finish")
        
        # train_size: 201*601 3s
        # test_size: 201*2001 10s
        
        return train_data, train_feature, train_spk_ids, test_data, test_feature, test_spk_ids

