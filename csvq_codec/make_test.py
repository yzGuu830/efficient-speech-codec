import matplotlib.pyplot as plt
import seaborn as sns
from config import cfg
from utils import process_control, load, collate, to_device
from data import input_collate
from models import autoencoder
from datasets import utils
from metrics.metrics import Metric
import torch
from IPython.display import Audio, display
import torchaudio
import os


def get_idx_data(data, data_idx, transform):
    input = {'audio':data[0][data_idx].unsqueeze(0), 'stft_feat':data[1][data_idx].unsqueeze(0), 'speaker_id':data[2][data_idx]}
    input['stft_feat'] = transform(input['stft_feat'])
    return input

def make_spec(input_feat, recon_feat, path, evals):
    '''input_feat: [1, F, T, 2] recon_feat: [1, F, T, 2]'''
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    # vis the amplitude domain
    input_feat = torch.view_as_complex(input_feat).squeeze(0) # [F, T]
    recon_feat = torch.view_as_complex(recon_feat).squeeze(0) # [F, T]
    
    input_feat  = 20*torch.log10(torch.clamp(input_feat.abs(),1e-5)) 
    recon_feat  = 20*torch.log10(torch.clamp(recon_feat.abs(),1e-5))
    
    fig.suptitle(str(evals))

    ax[0].set_title(f"input_feat_mag")
    ax[1].set_title(f"recon_feat_mag")

    sns.heatmap(input_feat.numpy(),ax = ax[0], cmap="mako", yticklabels=False).invert_yaxis()
    sns.heatmap(recon_feat.numpy(),ax = ax[1], cmap="mako", yticklabels=False).invert_yaxis()
    
    plt.show()
    fig.savefig(path, dpi=500, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    data_idx = 1
    print("Test data piece: ", data_idx)

    audio_data = load(f"/hpc/group/tarokhlab/yg172/data/{cfg['data_name']}/processed/test.pt",mode='torch')

    model = autoencoder.init_model()
    model = model.cuda()

    model_dict = '0_LIBRISPEECH_csvqcodec_nonscalable_18.0kbps_best.pt'
    model.load_state_dict(torch.load(f'/scratch/yg172/output/model/{model_dict}')['model_state_dict'])

    transform = utils.Compose([utils.Standardize(stats=cfg['stats'][cfg['data_name']])])

    input = get_idx_data(audio_data,data_idx,transform)
    input = to_device(input, cfg['device'])
    
    metric = Metric({'test': ['Loss','MEL_Loss','audio_PSNR','PESQ']})
    savepath = '/scratch/yg172/test_output/CSVQ_plot_audio_{}.jpg'.format(data_idx)

    with torch.no_grad():
        model.train(False)
        
        output = model(input)
        evaluation = metric.evaluate(metric.metric_name['test'], input, output)
        evals = {k:round(v,3) for k,v in evaluation.items()}
        print(evals)
        print()

        recon_feat = transform(output['recon_feat'].permute(0,2,3,1).contiguous()) #(1, F, T, C=2)
    
    # vis spec plot
    make_spec(input['stft_feat'].cpu(),recon_feat.cpu(), savepath, evals)
    
    # save reconstructed audio
    raw, recon = input['audio'].cpu(), output['recon_audio'].cpu()

    torchaudio.save('/scratch/yg172/test_output/audio{}_raw.flac'.format(data_idx),raw,16000)
    torchaudio.save('/scratch/yg172/test_output/audio{}_recon.flac'.format(data_idx),recon,16000)