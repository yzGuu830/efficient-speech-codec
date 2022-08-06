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



from scipy import signal
import numpy as np

class STFT:
    def __call__(self,x):
        stftx = signal.stft(x,fs=16000)[2]
        amp = np.abs(stftx)
        phase = np.angle(stftx)
        return amp, phase


class STDCT:
    def __call__(self,x):
        spectrogram = sdct(x, 1024, 256)
        return spectrogram


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input['data'] = t(input['data'])
            input['target'] = t(input['target'])
        return input

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string




# below is stdct module

import librosa
import numpy as np
import scipy


def sdct(signal, frame_length, frame_step, window="hamming"):
    """Compute Short-Time Discrete Cosine Transform of `signal`.
    No padding is applied to the signals.
    Parameters
    ----------
    signal : Time-domain input signal of shape `(n_samples,)`.
    frame_length : Window length and DCT frame length in samples.
    frame_step : Number of samples between adjacent DCT columns.
    window : Window specification passed to ``librosa.filters.get_window``.
        Default: "hamming".  Window to use for DCT.
    Returns
    -------
    dct : Real-valued F-T domain DCT matrix of shape `(frame_length, n_frames)`
    """
    framed = librosa.util.frame(signal, frame_length, frame_step)
    if window is not None:
        window = librosa.filters.get_window(window, frame_length, fftbins=True).astype(
            signal.dtype
        )
        framed = framed * window[:, np.newaxis]
    return scipy.fft.dct(framed, norm="ortho", axis=-2)


def isdct(dct, *, frame_step, frame_length=None, window="hamming"):
    """Compute Inverse Short-Time Discrete Cosine Transform of `dct`.
    Parameters other than `dct` are keyword-only.
    Parameters
    ----------
    dct : DCT matrix from `sdct`.
    frame_step : Number of samples between adjacent DCT columns (should be the
        same value that was passed to `sdct`).
    frame_length : Ignored. Window length and DCT frame length in samples.
        Can be None (default) or same value as passed to `sdct`.
    window : Window specification passed to ``librosa.filters.get_window``.
        Default: "hamming".  Window to use for IDCT.
    Returns
    -------
    signal : Time-domain signal reconstructed from `dct` of shape `(n_samples,)`.
        Note that `n_samples` may be different from the original signals' lengths as passed to `sdct_torch`,
        because no padding is applied.
    """
    frame_length2, n_frames = dct.shape[1], dct.shape[2]
    assert frame_length in {None, frame_length2}
    signal = overlap_add(
        scipy.fft.idct(dct, norm="ortho", axis=-2), frame_step=frame_step
    )
    if window is not None:
        window = librosa.filters.get_window(window, frame_length2, fftbins=True).astype(
            dct.dtype
        )
        window_frames = np.tile(window[:, np.newaxis], (1, n_frames))
        window_signal = overlap_add(window_frames, frame_step=frame_step)
        signal = signal / window_signal
    return signal


def overlap_add(framed, *, frame_step, frame_length=None):
    """Overlap-add ("deframe") a framed signal.
    Parameters other than `framed` are keyword-only.
    Parameters
    ----------
    framed : array_like of shape `(..., frame_length, n_frames)`.
    frame_step : Overlap to use when adding frames.
    Returns
    -------
    deframed : Overlap-add ("deframed") signal.
        np.ndarray of shape `(..., (n_frames - 1) * frame_step + frame_length)`.
    """
    *shape_rest, frame_length2, n_frames = framed.shape
    assert frame_length in {None, frame_length2}
    deframed_size = (n_frames - 1) * frame_step + frame_length2
    deframed = np.zeros((*shape_rest, deframed_size), dtype=framed.dtype)
    for i in range(n_frames):
        pos = i * frame_step
        deframed[..., pos : pos + frame_length2] += framed[..., i]
    return deframed