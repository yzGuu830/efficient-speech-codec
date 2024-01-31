"""script to consturct training/testing audio clips of LibriTTS (24kHz)"""

import torch, random, glob, os, torchaudio, tarfile, hashlib, gzip, zipfile
from random import shuffle
from tqdm import tqdm
from huggingface_hub import hf_hub_download

ROOT_PATH = "./"
SAVE_PATH = "./LibriTTS/processed_wav/"
DEFAULT_SR = 24000

def download_url(url, root, filename, md5):
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'pytorch/vision')]
    urllib.request.install_opener(opener)
    path = os.path.join(root, filename)
    if not os.path.exists(root): os.makedirs(root)
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

def check_integrity(path, md5=None):
    if not os.path.isfile(path):
        return False
    if md5 is None:
        return True
    return check_md5(path, md5)

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

def download(urls):
    for (url, md5) in urls:
        filename = os.path.basename(url)
        download_url(url, "./", filename, md5)
        extract_file(os.path.join("./", filename))

def download_extract_hf():    
    hf_hub_download(repo_id="Tracygu/LibriTTS-custom", 
            filename=f"test-clean.tar.gz", repo_type="dataset",
            cache_dir="./", local_dir="./", resume_download=True)
    extract_file("./test-clean.tar.gz", dest="./LibriTTS/test_raw/")

    hf_hub_download(repo_id="Tracygu/LibriTTS-custom", 
            filename=f"train-clean-360.tar.gz", repo_type="dataset",
            cache_dir="./", local_dir="./", resume_download=True)
    extract_file("./train-clean-360.tar.gz", dest="./LibriTTS/train_raw/")

def generate_audio_clips(audio_files, 
                         num_clips, 
                         clip_len, 
                         num_clips_per_subfolder,
                         split="test",
                         shuffle_seed=42,):

    random.seed(shuffle_seed)
    shuffle(audio_files)

    num_saved_wavs = 0
    cat_audio = torch.Tensor() # temp variable to hold current tensor to be cut, in case we need to piece more than one together
    for _, audio in tqdm(enumerate(audio_files), desc=f"contructing {split} split clips..."):

        if num_saved_wavs == num_clips:
            print(f"Found sufficiently {num_saved_wavs} {split} clips!")
            break

        wf, sr = torchaudio.load(audio) # returns waveform tensor and sampling rate --> also waveform tensor has shape (channel, num_samples)
        if sr != DEFAULT_SR:
            print(f"Found audio in {sr}Hz. Skip!")
            continue

        cat_audio = torch.cat((cat_audio, wf), dim=1)           # appended onto current audio clip with _ seconds
        if cat_audio.size(1) > int(clip_len * DEFAULT_SR):      # while the length is too long to be considered a sample, cut it and add clip
            
            clipped_audio = cat_audio[:, :int(clip_len * DEFAULT_SR)]
            if clipped_audio.abs().sum() > 1e-3:

                folder_num = num_saved_wavs // num_clips_per_subfolder + 1
                folder_name = f"{SAVE_PATH}/{split}/{(folder_num-1)*num_clips_per_subfolder+1}-{folder_num*num_clips_per_subfolder}"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                torchaudio.save(f'{folder_name}/clip_{num_saved_wavs+1}.wav', clipped_audio, DEFAULT_SR)     
                # cat_audio = torch.Tensor()
                cat_audio = cat_audio[:, int(clip_len * DEFAULT_SR):]  
                num_saved_wavs += 1
            else:
                print("Found a low-energy clip. Skip")
    
    if num_saved_wavs < num_clips: 
        print(f"Did not find enough clips for {split}")
        print(f"Only Got {num_saved_wavs}")

if __name__ == "__main__":

    download_extract_hf()

    # test_files = glob.glob(f"{ROOT_PATH}/test-clean/*/*/*.wav")
    # train_files = glob.glob(f"{ROOT_PATH}/train-clean/*/*/*.wav")

    # generate_audio_clips(test_files, 
    #                  num_clips=1500, 
    #                  clip_len=10, 
    #                  num_clips_per_subfolder=1500,
    #                  split="test",
    #                  shuffle_seed=42)
    

    # generate_audio_clips(train_files, 
    #                  num_clips=150000, 
    #                  clip_len=3, 
    #                  num_clips_per_subfolder=5000,
    #                  split="train",
    #                  shuffle_seed=42)