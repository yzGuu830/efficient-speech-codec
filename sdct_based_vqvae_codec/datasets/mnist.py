import anytree
import codecs
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index


class MNIST(Dataset):
    data_name = 'MNIST'
    file = [('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
             'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
             '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
             'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
             'ec29112dd5afa0611ce80d1b7f02629c')]

    def __init__(self, root, split, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)),
                                          mode='pickle')
        self.classes_counts = make_classes_counts(self.target)
        # self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        self.other = {'id': id}

    def __getitem__(self, index):
        data, target = Image.fromarray(self.data[index], mode='L'), torch.tensor(self.target[index])
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        input = {**other, 'data': data, 'target': target}
        if self.transform is not None:
            input['data'] = self.transform(input['data'])
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        # save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        train_data = read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte'))
        test_data = read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte'))
        train_target = read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        test_target = read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        # classes_to_labels = anytree.Node('U', index=[])
        # classes = list(map(str, list(range(10))))
        # for c in classes:
        #     make_tree(classes_to_labels, [c])
        # target_size = make_flat_index(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target)


class FashionMNIST(MNIST):
    data_name = 'FashionMNIST'
    file = [('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
             '8d4fb7e6c68d591d4c3dfef9ec88bf0d'),
            ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
             'bef4ecab320f06d8554ea6380940ec79'),
            ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
             '25c81989df183df01b3e8a0aad5dffbe'),
            ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
             'bb300cfdad3c16e7a12a480ee83cd310')]

    def make_data(self):
        train_data = read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte'))
        test_data = read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte'))
        train_target = read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        test_target = read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        classes_to_labels = anytree.Node('U', index=[])
        classes = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']
        for c in classes:
            make_tree(classes_to_labels, c)
        target_size = make_flat_index(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape((length, num_rows, num_cols))
        return parsed


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8).reshape(length).astype(np.int64)
        return parsed
