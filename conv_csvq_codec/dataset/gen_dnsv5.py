import glob
import torchaudio
import torchaudio.functional as F
from tqdm import tqdm
from random import shuffle
import torch
import os
import numpy as np
import pickle
import errno

TRAIN_CLIPS = 180000
TEST_CLIPS = 1158

TRAIN_LEN = 3
TEST_LEN = 10

DEFAULT_SR = 16e3

ROOT_PATH = '/hpc/group/tarokhlab/yg172/data/DNS_CHALLENGE/mnt/dnsv5/clean'

french_files = glob.glob(f'{ROOT_PATH}/french_speech/*/*/*/*/*/*/*.wav') # 48kHz
german_files = glob.glob(f'{ROOT_PATH}/german_speech/*/data/*.wav') + glob.glob(f'{ROOT_PATH}/german_speech/*/*.wav') # 48kHz
italian_files = glob.glob(f'{ROOT_PATH}/italian_speech/*/*/*/*/*/wavs/*.wav') # 48kHz
spanish_files = glob.glob(f'{ROOT_PATH}/spanish_speech/*.wav') # 48kHz
read_files = glob.glob(f'{ROOT_PATH}/read_speech/*.wav') # 48kHz

test_audio_files = glob.glob(f'/hpc/group/tarokhlab/yg172/data/DNS_CHALLENGE/V5_dev_testset/Track1_Headset/enrol/*.wav')
train_audio_files = french_files + german_files + italian_files + spanish_files + read_files

shuffle(train_audio_files)
shuffle(test_audio_files)

def main():
    if not check_exists("/hpc/group/tarokhlab/yg172/data/DNS_CHALLENGE/processed"):

        train_clips = gen_clips(train_audio_files, TRAIN_CLIPS, TRAIN_LEN)
        test_clips = gen_clips(test_audio_files, TEST_CLIPS, TEST_LEN)

        save(train_clips, "/hpc/group/tarokhlab/yg172/data/DNS_CHALLENGE/processed/train.pt", mode='torch')
        save(test_clips, "/hpc/group/tarokhlab/yg172/data/DNS_CHALLENGE/processed/test.pt", mode='torch')

def gen_clips(audio_files, num_clips, clip_len):
    clips = []
    cat_audio = torch.Tensor()

    for audio in tqdm(audio_files):

        if len(clips) >= num_clips:
            print(f"Got all {num_clips} train clips")
            break

        wf, sr = torchaudio.load(audio)

        if sr != DEFAULT_SR:  wf = resample(wf, sr, DEFAULT_SR) # resample to 16e3

        cat_audio = torch.cat((cat_audio, wf), dim=1)

        while cat_audio.size(1) > int(clip_len * DEFAULT_SR):

            clips.append(cat_audio[:, :int(clip_len * DEFAULT_SR)])

            cat_audio = cat_audio[:, int(clip_len * DEFAULT_SR):] # cut clip

    return torch.cat(clips, dim=0) # (TRAIN_CLIPS, 48e3) (TEST_CLIPS, 16e4)


def resample(waveform, sr, tg_sr):
    # https://pytorch.org/tutorials/beginner/audio_resampling_tutorial.html
    return F.resample(waveform, sr, tg_sr, rolloff=0.98)

def check_exists(path): # check if the path exists
    return os.path.exists(path)

def makedir_exist_ok(path): # make a directory path
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return

def save(input, path, mode='torch'): # save data
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        pickle.dump(input, open(path, 'wb'))
    else:
        raise ValueError('Not valid save mode')
    return


if __name__ == "__main__":

    # print('Num of Utterances: ', len(french_files), len(german_files), len(italian_files), len(spanish_files), len(read_files))

    # french, german, italian, spanish, read = torchaudio.load(french_files[10]), torchaudio.load(german_files[10]), torchaudio.load(italian_files[10]), torchaudio.load(spanish_files[10]), torchaudio.load(read_files[10]) 
    # print('SR: ', french[1], german[1], italian[1], spanish[1], read[1])

    # print('Duration: ', french[0].size(1)/french[1], german[0].size(1)/german[1], italian[0].size(1)/italian[1], spanish[0].size(1)/spanish[1], read[0].size(1)/read[1])

    main()

    # test_clips = gen_clips(test_audio_files, TEST_CLIPS, TEST_LEN)
    # save(test_clips, "/hpc/group/tarokhlab/yg172/data/DNS_CHALLENGE/processed/test.pt", mode='torch')

