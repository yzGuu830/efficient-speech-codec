from config import cfg
import torch
import torchaudio
import datasets
import torchvision

def make_transform(mode):
    if mode == 'plain':
        transform = make_plain_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
    elif mode == 'basic':
        transform = make_basic_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
    elif mode == 'basic-spec':
        transform = make_basic_spec_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
    elif mode == 'basic-rand':
        transform = make_basic_rand_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
    elif mode == 'basic-rands':
        transform = make_basic_rands_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
    elif mode == 'basic-spec-rands':
        transform = make_basic_spec_rands_transform(cfg['data_length'], cfg['n_fft'], cfg['hop_length'])
    elif 'fix' in mode:
        transform = make_fix_transform(cfg['data_name'])
    else:
        raise ValueError('Not valid aug')
    return transform


def make_fix_transform(data_name):
    transform = FixTransform(data_name)
    return transform


def make_plain_transform(data_length, n_fft, hop_length):
    plain_transform = [datasets.transforms.CenterCropPad(data_length),
                       torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=40),
                       torchaudio.transforms.AmplitudeToDB('power', 80),
                       datasets.transforms.SpectoImage(),
                       torchvision.transforms.ToTensor()]
    plain_transform = torchvision.transforms.Compose(plain_transform)
    return plain_transform


def make_basic_transform(data_length, n_fft, hop_length):
    basic_transform = [datasets.transforms.RandomTimeResample([0.85, 1.15]),
                       datasets.transforms.CenterCropPad(data_length),
                       datasets.transforms.RandomTimeShift(0.1),
                       torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=40),
                       torchaudio.transforms.AmplitudeToDB('power', 80),
                       datasets.transforms.SpectoImage(),
                       torchvision.transforms.ToTensor()]
    basic_transform = torchvision.transforms.Compose(basic_transform)
    return basic_transform


def make_basic_spec_transform(data_length, n_fft, hop_length):
    basic_spec_transform = [datasets.transforms.RandomTimeResample([0.85, 1.15]),
                            datasets.transforms.CenterCropPad(data_length),
                            datasets.transforms.RandomTimeShift(0.1),
                            torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=40),
                            torchaudio.transforms.FrequencyMasking(7),
                            torchaudio.transforms.TimeMasking(12),
                            torchaudio.transforms.AmplitudeToDB('power', 80),
                            datasets.transforms.SpectoImage(),
                            torchvision.transforms.ToTensor()]
    basic_spec_transform = torchvision.transforms.Compose(basic_spec_transform)
    return basic_spec_transform


def make_basic_rand_transform(data_length, n_fft, hop_length):
    basic_rand_transform = [datasets.transforms.RandomTimeResample([0.85, 1.15]),
                            datasets.transforms.CenterCropPad(data_length),
                            datasets.transforms.RandomTimeShift(0.1),
                            torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=40),
                            torchaudio.transforms.AmplitudeToDB('power', 80),
                            datasets.transforms.SpectoImage(),
                            datasets.randaugment.RandAugment(n=2, m=10),
                            torchvision.transforms.ToTensor()]
    basic_rand_transform = torchvision.transforms.Compose(basic_rand_transform)
    return basic_rand_transform


def make_basic_rands_transform(data_length, n_fft, hop_length):
    basic_rands_transform = [datasets.transforms.RandomTimeResample([0.85, 1.15]),
                             datasets.transforms.CenterCropPad(data_length),
                             datasets.transforms.RandomTimeShift(0.1),
                             torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=40),
                             torchaudio.transforms.AmplitudeToDB('power', 80),
                             datasets.transforms.SpectoImage(),
                             datasets.randaugment.RandAugmentSelected(n=2, m=10),
                             torchvision.transforms.ToTensor()]
    basic_rands_transform = torchvision.transforms.Compose(basic_rands_transform)
    return basic_rands_transform


def make_basic_spec_rands_transform(data_length, n_fft, hop_length):
    basic_spec_rands_transform = [datasets.transforms.RandomTimeResample([0.85, 1.15]),
                                  datasets.transforms.CenterCropPad(data_length),
                                  datasets.transforms.RandomTimeShift(0.1),
                                  torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length=hop_length, n_mels=40),
                                  torchaudio.transforms.FrequencyMasking(7),
                                  torchaudio.transforms.TimeMasking(12),
                                  torchaudio.transforms.AmplitudeToDB('power', 80),
                                  datasets.transforms.SpectoImage(),
                                  datasets.randaugment.RandAugmentSelected(n=2, m=10),
                                  torchvision.transforms.ToTensor()]
    basic_spec_rands_transform = torchvision.transforms.Compose(basic_spec_rands_transform)
    return basic_spec_rands_transform

class FixTransform(torch.nn.Module):
    def __init__(self, data_name):
        super().__init__()
        self.weak = datasets.Compose(
            [make_transform(cfg['sup_aug']), torchvision.transforms.Normalize(*cfg['stats'][data_name])])
        self.strong = datasets.Compose(
            [make_transform(cfg['unsup_aug']), torchvision.transforms.Normalize(*cfg['stats'][data_name])])

    def forward(self, input):
        data = self.weak({'data': input['data']})['data']
        aug_data = self.strong({'data': input['data']})['data']
        input = {**input, 'data': data, 'aug_data': aug_data}
        return input