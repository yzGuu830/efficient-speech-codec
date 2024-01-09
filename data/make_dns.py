import glob
import torchaudio
import torchaudio.functional as F
from tqdm import tqdm
import random
from random import shuffle
import torch
import os
import numpy as np
import pickle
import errno

"""
Compose a Train/Test audio dataset 
    make sure all audios are 16kHz; 
    make sure train/test speakers are different; 
    make sure each audio is cutted at most once (and only the beginning piece is used, the remaining is discarded; For longer samples, compose multiple source audios)

Train: 120000 samples in English / 60000 samples in other language (3 sec) -> Tensor (180000, 16k*3)
Test: 772 samples in English / 386 samples in other language (10 sec)      -> Tensor (1158, 16k*10)
Train/Test samples -> train.pt / test.pt ; Save all test samples into wav files (tagged with index, e.g., test_instance1.wav)

Instructions
- source data is at /scratch/eys9/data/DNS_CHALLENGE/datasets/clean (on machine 10) (ssh eys9@research-tarokhlab-10.oit.duke.edu)
  Folder Structure
    - english [61G] (all 16kHz already known) (all containing speaker information in the file name)
    - french [21G] 
    - german [66G]
    - italian [14G]
    - mandarian [21G]
    - russian [5.1G]
    - spanish [17G]
- First Iterate the dataset and obtain some information:
    for other language (may contain 48kHz audios, print sample rate distribution first, refer to func: make_info)
    for other language (speaker info may/may not be contained, see the folder structure)
- Split Train/Test
- Save Testset into wave files
"""

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

TRAIN_CLIPS = 180000
TRAIN_ENG_CLIPS = 120000
TRAIN_MULTI_CLIPS = 10000

TEST_CLIPS = 1158
TEST_ENG_CLIPS = 774
TEST_MULTI_CLIPS = 64

TRAIN_LEN = 3
TEST_LEN = 10

DEFAULT_SR = 16000

ROOT_PATH = '/scratch/eys9/data/DNS_CHALLENGE/datasets/clean'
SAVE_PATH = '/scratch/eys9/data/DNS_CHALLENGE'
makedir_exist_ok(f"{SAVE_PATH}/raw")

# multilengual root paths
FRENCH_ROOT = '/french_data/M-AILABS_Speech_Dataset/fr_FR_190hrs_16k'
GERMAN_ROOT = '/german_speech/CC_BY_SA_4.0_Forschergeist_2hours_2spk_German_16k/data'
ITALIAN_ROOT = '/italian_speech/M-AILABS_Speech_Dataset/it_IT_128hrs_16k'
SPANISH_ROOT = '/spanish_speech/M-AILABS_Speech_Dataset/es_ES_16k'
MANDARIN_ROOT = '/mandarin_speech/SLR18_THCHS00'
RUSSIAN_ROOT = '/russian_speech/M-AILABS_Speech_Dataset/ru_RU_47hrs_16k'

# glob.glob returns a list of the files paths that matches with the pattern, so just like terminal
french_train_files =  glob.glob(f'{ROOT_PATH}{FRENCH_ROOT}/female/ezwa/keraban_le_tetu/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{FRENCH_ROOT}/female/ezwa/la_fabrique_de_crimes/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{FRENCH_ROOT}/female/ezwa/la_mare_au_diable/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{FRENCH_ROOT}/female/ezwa/l_epouvante/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{FRENCH_ROOT}/female/ezwa/monsieur_lecoq/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{FRENCH_ROOT}/male/bernard/le_dernier_des_mohicans/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{FRENCH_ROOT}/male/bernard/le_pays_des_fourrures/wavs/*.wav')
french_test_files =  glob.glob(f'{ROOT_PATH}{FRENCH_ROOT}/mix/sous_les_mers/wavs/*.wav')

german_train_files = glob.glob(f'{ROOT_PATH}/german_speech/CC_BY_SA_4.0_249hrs_339spk_German_Wikipedia_16k/data/*.wav') 
german_test_files = glob.glob(f'{ROOT_PATH}{GERMAN_ROOT}/*.wav')

italian_train_files = glob.glob(f'{ROOT_PATH}{ITALIAN_ROOT}/female/lisa_caputo/malavoglia/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{ITALIAN_ROOT}/male/riccardo_fasol/galatea/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{ITALIAN_ROOT}/male/riccardo_fasol/il_ritratto_del_diavolo/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{ITALIAN_ROOT}/male/riccardo_fasol/la_contessa_di_karolystria/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{ITALIAN_ROOT}/male/riccardo_fasol/il_fu_mattia_pascal/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{ITALIAN_ROOT}/male/riccardo_fasol/le_meraviglie_del_duemila/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{ITALIAN_ROOT}/mix/novelle_per_un_anno_00/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{ITALIAN_ROOT}/mix/novelle_per_un_anno_01/wavs/*.wav')
italian_test_files =  glob.glob(f'{ROOT_PATH}{ITALIAN_ROOT}/mix/novelle_per_un_anno_02/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{ITALIAN_ROOT}/mix/novelle_per_un_anno_03/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{ITALIAN_ROOT}/mix/novelle_per_un_anno_04/wavs/*.wav')
 
spanish_files = glob.glob(f'{ROOT_PATH}{SPANISH_ROOT}/*.wav')
spanish_train_files = spanish_files[int(len(spanish_files)*0.1):]
spanish_test_files = spanish_files[:int(len(spanish_files)*0.1)]

mandarin_train_files = glob.glob(f'{ROOT_PATH}/mandarin_speech/slr33_aishell/S0002/*.wav')
for i in range(100, 364):
    mandarin_train_files += glob.glob(f'{ROOT_PATH}/mandarin_speech/slr33_aishell/S0{i}/*.wav')
mandarin_test_files = glob.glob(f'{ROOT_PATH}{MANDARIN_ROOT}/*.wav')

russian_train_files = glob.glob(f'{ROOT_PATH}{RUSSIAN_ROOT}/female/hajdurova/chetvero_nischih/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{RUSSIAN_ROOT}/female/hajdurova/jizn_gnora/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{RUSSIAN_ROOT}/female/hajdurova/strashnaya_minuta/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{RUSSIAN_ROOT}/female/hajdurova/falshiviy_kupon/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{RUSSIAN_ROOT}/female/hajdurova/mnogo_shuma_iz_nichego/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{RUSSIAN_ROOT}/male/minaev/oblomov/wavs/*.wav')
russian_test_files =  glob.glob(f'{ROOT_PATH}{RUSSIAN_ROOT}/female/hajdurova/sulamif/wavs/*.wav') \
                    + glob.glob(f'{ROOT_PATH}{RUSSIAN_ROOT}/male/nikolaev/argentinetz/wavs/*.wav')

english_files = glob.glob(f'{ROOT_PATH}/read_speech/*.wav') 
english_train_files = english_files[int(len(english_files)*0.1):]
english_test_files = english_files[:int(len(english_files)*0.1)]


# 2. Audio Composing (Needs to be modified, should make sure each audio is cutted at most once)
def gen_clips(audio_files, num_clips, clip_len, test=False, tag="english"):

    random.seed(0)
    shuffle(audio_files)

    clips = []
    cat_audio = torch.Tensor() # temp variable to hold current tensor to be cut, in case we need to piece more than one together
    if test:
        makedir_exist_ok(f"{SAVE_PATH}/raw/test_audios")

    for _, audio in tqdm(enumerate(audio_files)): # displays a progress bar

        if len(clips) >= num_clips:
            break

        wf, _ = torchaudio.load(audio) # returns waveform tensor and sampling rate --> also waveform tensor has shape (channel, num_samples)

        cat_audio = torch.cat((cat_audio, wf), dim=1)           # appended onto current audio clip with _ seconds
        
        if tag in ["german", "spanish", "english"] and not test:
            # different way to treat german train audios
            while cat_audio.size(1) > int(clip_len * DEFAULT_SR):
                clipped_audio = cat_audio[:, :int(clip_len * DEFAULT_SR)]
                clips.append(clipped_audio)

                cat_audio = cat_audio[:, int(clip_len * DEFAULT_SR):]
                
            cat_audio = torch.Tensor()

        else:
            if cat_audio.size(1) > int(clip_len * DEFAULT_SR):      # while the length is too long to be considered a sample, cut it and add clip

                clipped_audio = cat_audio[:, :int(clip_len * DEFAULT_SR)]
                clips.append(clipped_audio)

                if test:
                    idx = len(clips)
                    torchaudio.save(f'{SAVE_PATH}/raw/test_audios/{tag}_instance{idx}.wav', clipped_audio, DEFAULT_SR)      

                cat_audio = torch.Tensor()
                # cat_audio = cat_audio[:, int(clip_len * DEFAULT_SR):]  
            else:
                continue           

        # cat_audio = torch.Tensor()
    
    if len(clips) < num_clips: 
        print(f"Did not find Enough Samples for {tag}")
        print(f"Only Got {len(clips)}")

    return clips 
    
    # [num_clips, clip_len*DEFAULT_SR] 
    # remember to think of matrix here! num_clip rows, each with an audio clip length











def save(input, path, mode='torch'): # save data
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        pickle.dump(input, open(path, 'wb'))       # research: what is pickle?? 
    else:
        raise ValueError('Not valid save mode')
    return











# Main Function (Needs to Modify)

def main():

    print(f"Constucting {TEST_CLIPS} Testing Clips")
    test_clip_list = gen_clips(english_test_files, TEST_ENG_CLIPS, TEST_LEN, test=True, tag="english") + \
                    gen_clips(french_test_files, TEST_MULTI_CLIPS, TEST_LEN, test=True, tag="french") + \
                    gen_clips(german_test_files, TEST_MULTI_CLIPS, TEST_LEN, test=True, tag="german") + \
                    gen_clips(italian_test_files, TEST_MULTI_CLIPS, TEST_LEN, test=True, tag="italian") + \
                    gen_clips(spanish_test_files, TEST_MULTI_CLIPS, TEST_LEN, test=True, tag="spanish") + \
                    gen_clips(mandarin_test_files, TEST_MULTI_CLIPS, TEST_LEN, test=True, tag="mandarin") + \
                    gen_clips(russian_test_files, TEST_MULTI_CLIPS, TEST_LEN, test=True, tag="russian")
    test_all_clips = torch.cat(test_clip_list, dim=0)
    print("Test: ", test_all_clips.shape)
    save(test_all_clips, f'{SAVE_PATH}/processed_yz/test.pt', mode='torch')

    print(f"Constucting {TRAIN_CLIPS} Training Clips")
    train_clip_list = gen_clips(english_train_files, TRAIN_ENG_CLIPS, TRAIN_LEN, test=False, tag="english") + \
                    gen_clips(french_train_files, TRAIN_MULTI_CLIPS, TRAIN_LEN, test=False, tag="french") + \
                    gen_clips(german_train_files, TRAIN_MULTI_CLIPS, TRAIN_LEN, test=False, tag="german") + \
                    gen_clips(italian_train_files, TRAIN_MULTI_CLIPS, TRAIN_LEN, test=False, tag="italian") + \
                    gen_clips(spanish_train_files, TRAIN_MULTI_CLIPS, TRAIN_LEN, test=False, tag="spanish") + \
                    gen_clips(mandarin_train_files, TRAIN_MULTI_CLIPS, TRAIN_LEN, test=False, tag="mandarin") + \
                    gen_clips(russian_train_files, TRAIN_MULTI_CLIPS, TRAIN_LEN, test=False, tag="russian")

    train_all_clips = torch.cat(train_clip_list, dim=0)
    print("Train: ", train_all_clips.shape)
    save(train_all_clips, f'{SAVE_PATH}/processed_yz/train.pt', mode='torch')

    return





if __name__ == "__main__":
    main()

    

