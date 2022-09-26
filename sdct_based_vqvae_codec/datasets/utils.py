# import anytree
import hashlib
import os
import glob
import gzip
import tarfile
import zipfile
import numpy as np

from tqdm import tqdm
from collections import Counter
from utils import makedir_exist_ok
from torch_dct import sdct_torch, isdct_torch
import torch.nn.functional as F
import librosa
import torch
import torchvision

def make_classes_counts(label):
    label = np.array(label)
    if label.ndim > 1:
        label = label.sum(axis=tuple([i for i in range(1, label.ndim)]))
    classes_counts = Counter(label)
    return classes_counts

def make_bar_updater(pbar):
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(path, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(path, md5, **kwargs):
    return md5 == calculate_md5(path, **kwargs)


def check_integrity(path, md5=None):
    if not os.path.isfile(path):
        return False
    if md5 is None:
        return True
    return check_md5(path, md5)



def download_url(url, root, filename, md5):
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'pytorch/vision')]
    urllib.request.install_opener(opener)
    path = os.path.join(root, filename)
    makedir_exist_ok(root)
    if os.path.isfile(path) and check_integrity(path, md5):
        print('Using downloaded and verified file: ' + path)
    else:
        try:
            print('Downloading ' + url + ' to ' + path)
            urllib.request.urlretrieve(url, path, reporthook=make_bar_updater(tqdm(unit='B', unit_scale=True)))
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + path)
                urllib.request.urlretrieve(url, path, reporthook=make_bar_updater(tqdm(unit='B', unit_scale=True)))
        if not check_integrity(path, md5):
            raise RuntimeError('Not valid downloaded file')
    return


def extract_file(src, dest=None, delete=False):
    print('Extracting {}'.format(src))
    dest = os.path.dirname(src) if dest is None else dest
    filename = os.path.basename(src)
    if filename.endswith('.zip'):
        with zipfile.ZipFile(src, "r") as zip_f:
            zip_f.extractall(dest)
    elif filename.endswith('.tar'):
        with tarfile.open(src) as tar_f:
            tar_f.extractall(dest)
    elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
        with tarfile.open(src, 'r:gz') as tar_f:
            tar_f.extractall(dest)
    elif filename.endswith('.gz'):
        with open(src.replace('.gz', ''), 'wb') as out_f, gzip.GzipFile(src) as zip_f:
            out_f.write(zip_f.read())
    if delete:
        os.remove(src)
    return

def make_tree(root, name, attribute=None):
    if len(name) == 0:
        return
    if attribute is None:
        attribute = {}
    this_name = name[0]
    next_name = name[1:]
    this_attribute = {k: attribute[k][0] for k in attribute}
    next_attribute = {k: attribute[k][1:] for k in attribute}
    this_node = anytree.find_by_attr(root, this_name)
    this_index = root.index + [len(root.children)]
    if this_node is None:
        this_node = anytree.Node(this_name, parent=root, index=this_index, **this_attribute)
    make_tree(this_node, next_name, next_attribute)
    return


def make_flat_index(root, given=None):
    if given:
        classes_size = 0
        for node in anytree.PreOrderIter(root):
            if len(node.children) == 0:
                node.flat_index = given.index(node.name)
                classes_size = given.index(node.name) + 1 if given.index(node.name) + 1 > classes_size else classes_size
    else:
        classes_size = 0
        for node in anytree.PreOrderIter(root):
            if len(node.children) == 0:
                node.flat_index = classes_size
                classes_size += 1
    return classes_size


from scipy import signal
import numpy as np

class STFT:
    def __call__(self,x):
        stftx = signal.stft(x,fs=16000)[2]
        amp = np.abs(stftx)
        phase = np.angle(stftx)
        return amp, phase

class STDCT:
    ## frame_size: 6.25ms hop_size: 3.75ms
    def __init__(self,frame_length=120,hop_length=60):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def __call__(self,x):
        if isinstance(x,dict):
            spectrogram = sdct_torch(x['audio'], frame_length=self.frame_length, frame_step=self.hop_length)
        else:
            spectrogram = sdct_torch(x, frame_length=self.frame_length, frame_step=self.hop_length)
        # 1*120*136
        return spectrogram

class PowerSpectrogram:
    def __call__(self,x):
        return 10 * torch.log10(torch.clamp(torch.pow(x,2), min=1e-10))

def get_sign(x):
    sign = torch.ones(x.shape)
    sign = sign.masked_fill((x<0),-1)
    return sign

class InversePowerSpectrogram:
    def __call__(self,x):
        return torch.pow(torch.pow(10.0, 0.1 * x), 0.5)

class Standardize:
    def __call__(self,x): # x is stdct feature
        # x = x.to(torch.float64) 
        # print(torch.tensor([x.mean().item(), x.std().item()]))
        return torch.tensor([x.mean().item(), x.std().item()]), normalize(x)

def normalize(input):
    broadcast_size = [1] * input.dim()
    # broadcast_size[1] = input.size(1)
    m, s = input.mean().item(), input.std().item()
    # print(input,input.mean(),input.std())
    # if s == 0: print("FUCKKKKKK")
    m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
           torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
    input = input.sub(m).div(s)
    return input


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        if self.transform_name() == "Compose( Standardize PowerSpectrogram Standardize )":
            sdct_stats, standardized_sdct = self.transforms[0](input)
            power_feat = self.transforms[1](standardized_sdct)
            power_stats, standardized_power = self.transforms[2](power_feat)
            return {'power_feat':standardized_power,'power_stats':power_stats,'sdct_stats':sdct_stats}
        elif self.transform_name() == "Compose( Standardize PowerSpectrogram )":
            sdct_stats, standardized_sdct = self.transforms[0](input)
            power_feat = self.transforms[1](standardized_sdct)
            return {'power_feat':power_feat,'sdct_stats':sdct_stats}
        elif self.transform_name() == "Compose( PowerSpectrogram )":
            power_feat = self.transforms[0](input)
            return {'power_feat':power_feat}
        else:
            raise ValueError('Not valid transform composition')
            
    def __repr__(self):
        return self.transform_name()

    def transform_name(self):
        format_string = self.__class__.__name__ + '( '
        for t in self.transforms:
            format_string += '{0} '.format(t.__class__.__name__)
        format_string += ')'
        return format_string


if __name__ == "__main__":
    def get_feature(audios):
        feat, sign, stats = [], [], []
        for audio in audios:
            x = STDCT()(audio)
            output = normalize()(x)
            normalized_x, data_stats = output['normalized_x'],output['data_stats']
            stats.append(data_stats)
            output = PowerSpectrogram()(normalized_x)
            feat.append(output['x_db'])
            sign.append(output['sign'])
        return torch.stack(feat), torch.stack(sign), torch.stack(stats)
    data = torch.randn(4,8000)
    feature,sign,stats = get_feature(data)
