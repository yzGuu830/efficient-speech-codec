import matplotlib.pyplot as plt
import seaborn as sns
from metrics.metrics import audio_MSE,audio_PSNR,img_MSE,img_PSNR,PESQ
from config import cfg
from utils import process_control, load, collate
from data import input_collate
from models.vqvae import vqvae
from datasets import utils
import torch
from IPython.display import Audio, display
import torchaudio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def plot_spectrogram1(raw_feat, recon_feat, path):
    raw_feat, recon_feat = raw_feat.view(raw_feat.size()[-2:]), recon_feat.view(recon_feat.size()[-2:])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].set_title("raw_feat")
    ax[1].set_title("recon_feat")
    sns.heatmap(raw_feat.numpy(),ax = ax[0], cmap="mako", yticklabels=False).invert_yaxis()
    ax[0].set_ylabel("Frequency(Hz)")
    sns.heatmap(recon_feat.numpy(),ax = ax[1], cmap="mako", yticklabels=False).invert_yaxis()
    ax[1].set_ylabel("Frequency(Hz)")

    plt.show()
    fig.savefig(path, dpi=500, bbox_inches='tight', pad_inches=0)

def get_idx_data(data,idx,transform):
    input = {'audio':data[0][idx], 'sdct_feat':data[1][idx], 'sign':data[2][idx]}
    
    output = transform(input['sdct_feat'])

    for key in output.keys():
        input[key] = output[key]

    for key in input:
        input[key] = [input[key]]
    input['feature'] = input.pop('power_feat')
    
    return input

def inference(model,input,metric):
    with torch.no_grad():
        model.train(False)
        input = collate(input)
        output = model(input)
        for key in metric.keys():
            print('{}: {:.4f} '.format(key,metric[key](input,output)),end='')
        print()
        return output


def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")

def run_test(data_idx):
    process_control()
    cfg['seed'] = 0

    data = load("/hpc/group/tarokhlab/yg172/data/LIBRISPEECH/processed/test.pt",mode='torch')
    model = vqvae()
    model.load_state_dict(torch.load('/scratch/yg172/output/model/0_vqgan_based_codec_17.8k_test_best.pt',map_location=torch.device('cpu'))['model_state_dict'])
    metric = {'Loss': (lambda input, output: output['loss'].item()),
        'audio_MSE':  (lambda input, output: audio_MSE(output['recon_audio'], input['audio'])),
        'audio_PSNR': (lambda input, output: audio_PSNR(output['recon_audio'], input['audio'])),
        'img_MSE': (lambda input, output: img_MSE(output['recon_feature'], input['feature'])),
        'img_PSNR': (lambda input, output: img_PSNR(output['recon_feature'], input['feature'])),
        'PESQ': (lambda input, output: PESQ(output['recon_audio'], input['audio']))}
    transform = utils.Compose([ utils.Standardize(),
                                utils.PowerSpectrogram(),
                                utils.Standardize()])

    input = get_idx_data(data,data_idx,transform)
    print("Test_data_piece{}:".format(data_idx))
    output = inference(model,input,metric)
    
    PATH = '/scratch/yg172/test_output/testaudio{}.jpg'.format(data_idx)
    plot_spectrogram1(raw_feat=input['feature'],recon_feat=output['recon_feature'],path=PATH)

    raw, recon = input['audio'][0], output['recon_audio'][0]
    torchaudio.save('/scratch/yg172/test_output/testaudio{}_raw.flac'.format(data_idx),raw,16000)
    torchaudio.save('/scratch/yg172/test_output/testaudio{}_recon.flac'.format(data_idx),raw,16000)

    return 

if __name__ == "__main__":
    run_test(3)
    
    # play_audio(raw,16000)
    # play_audio(recon,16000)

    
    
    

    
