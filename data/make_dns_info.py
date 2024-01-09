import glob
import torchaudio
from tqdm import tqdm

ROOT_PATH = '/scratch/eys9/data/DNS_CHALLENGE/datasets/clean'
SAVE_PATH = '/scratch/eys9/data/DNS_CHALLENGE/raw/'

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

def make_info(files):

    num_non_16ks, num_16ks = 0, 0
    hrs = 0

    for file in tqdm(files):
        wf, sr = torchaudio.load(file)

        if sr != 16000:
            num_non_16ks += 1
            continue

        else:
            num_16ks += 1
            hrs += (wf.size(1) / sr) / 3600

    return hrs, num_16ks, num_non_16ks

if __name__ == "__main__":

    print("Num Audios: ", len(english_files))
    hrs, num_16ks, num_non_16ks = make_info(english_train_files)
    print(f"English Train Files: {hrs} hrs  ||  {num_16ks} 16k audio files  ||  {num_non_16ks} non16k audio files")
    hrs, num_16ks, num_non_16ks = make_info(english_test_files)
    print(f"English Test Files: {hrs} hrs  ||  {num_16ks} 16k audio files  ||  {num_non_16ks} non16k audio files")

    print("Num Audios: ", len(french_train_files) + len(french_test_files))
    hrs, num_16ks, num_non_16ks = make_info(french_train_files)
    print(f"French Train Files: {hrs} hrs  ||  {num_16ks} 16k audio files  ||  {num_non_16ks} non16k audio files")
    hrs, num_16ks, num_non_16ks = make_info(french_test_files)
    print(f"French Test Files: {hrs} hrs  ||  {num_16ks} 16k audio files  ||  {num_non_16ks} non16k audio files")

    print("Num Audios: ", len(german_train_files) + len(german_test_files))
    hrs, num_16ks, num_non_16ks = make_info(german_train_files)
    print(f"German Train Files: {hrs} hrs  ||  {num_16ks} 16k audio files  ||  {num_non_16ks} non16k audio files")
    hrs, num_16ks, num_non_16ks = make_info(german_test_files)
    print(f"German Test Files: {hrs} hrs  ||  {num_16ks} 16k audio files  ||  {num_non_16ks} non16k audio files")

    print("Num Audios: ", len(italian_train_files) + len(italian_test_files))
    hrs, num_16ks, num_non_16ks = make_info(italian_train_files)
    print(f"Italian Train Files: {hrs} hrs  ||  {num_16ks} 16k audio files  ||  {num_non_16ks} non16k audio files")
    hrs, num_16ks, num_non_16ks = make_info(italian_test_files)
    print(f"Italian Test Files: {hrs} hrs  ||  {num_16ks} 16k audio files  ||  {num_non_16ks} non16k audio files")

    print("Num Audios: ", len(spanish_files))
    hrs, num_16ks, num_non_16ks = make_info(spanish_train_files)
    print(f"Spanish Train Files: {hrs} hrs  ||  {num_16ks} 16k audio files  ||  {num_non_16ks} non16k audio files")
    hrs, num_16ks, num_non_16ks = make_info(spanish_test_files)
    print(f"Spanish Test Files: {hrs} hrs  ||  {num_16ks} 16k audio files  ||  {num_non_16ks} non16k audio files")

    print("Num Audios: ", len(mandarin_train_files) + len(mandarin_test_files))
    hrs, num_16ks, num_non_16ks = make_info(mandarin_train_files)
    print(f"Mandarin Train Files: {hrs} hrs  ||  {num_16ks} 16k audio files  ||  {num_non_16ks} non16k audio files")
    hrs, num_16ks, num_non_16ks = make_info(mandarin_test_files)
    print(f"Mandarin Test Files: {hrs} hrs  ||  {num_16ks} 16k audio files  ||  {num_non_16ks} non16k audio files")

    print("Num Audios: ", len(russian_train_files)+len(russian_test_files))
    hrs, num_16ks, num_non_16ks = make_info(russian_train_files)
    print(f"Russian Train Files: {hrs} hrs  ||  {num_16ks} 16k audio files  ||  {num_non_16ks} non16k audio files")
    hrs, num_16ks, num_non_16ks = make_info(russian_test_files)
    print(f"Russian Test Files: {hrs} hrs  ||  {num_16ks} 16k audio files  ||  {num_non_16ks} non16k audio files")