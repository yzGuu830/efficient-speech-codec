import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils import to_device, resume, plot_spectrogram
from config import cfg

from data import fetch_dataset, make_data_loader
from models.autoencoder import init_model
from metrics.metrics import Metric

import math
from tqdm import tqdm
import numpy as np
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

cfg['model_tag'] = '0_{}_csvqcodec_{}_{}kbps'.format('DNS_CHALLENGE',cfg['model_mode'], 18000*0.001)
cfg['seed'] = 0
torch.manual_seed(cfg['seed'])
torch.cuda.manual_seed(cfg['seed'])

def main():
    # load_dataset
    dataset = fetch_dataset(cfg['data_name'])
    data_loader = make_data_loader(dataset, cfg['model_name'])
    print(f"train_data:{len(data_loader['train'])} test_data:{len(data_loader['test'])}")

    # load_model
    model = nn.DataParallel(init_model())
    model = model.cuda()

    metric = Metric({'train': ['Loss','RECON_Loss','CODEBOOK_Loss','MEL_Loss'], 
                    'test': ['audio_PSNR','PESQ']})

    result = resume(cfg['model_tag'],'best')
        
    model.load_state_dict(result['model_state_dict'])

    for i in range(1,7):
        test_csvqcodec(data_loader['test'], model, metric, N=i)

def test_csvqcodec(data_loader, model, metric, N=6, save_image=cfg['save_image']):

    with torch.no_grad():
        model.train(False)
        plot_batch = {}

        res = {'audio_PSNR':[], 'PESQ':[]}
        for i, input in tqdm(enumerate(data_loader)):
            input = to_device(input, cfg['device'])

            output = model(**dict(input=input, train=False, target_Bs=N))   #fix the highest bitrate for test inference

            if i == 2: 
                plot_batch['raw_feat'] = input['stft_feat'][:4]
                plot_batch['recon_feat'] = output['recon_feat'][:4]
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)

            for key in evaluation:
                res[key].append(evaluation[key])

            if i*cfg['csvq_codec']['batch_size']['test'] >= 1158: break
        
        for key in res:
            res[key] = np.mean(res[key])

        print(f"Performance at {N*3}kbps: ", res)
        if save_image:
            img_path = f"/scratch/yg172/test/test_{N*3}kbps.jpg"
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)

            plot_spectrogram(plot_batch['raw_feat'],plot_batch['recon_feat'],img_path,evaluation,Bs=N)

if __name__ == '__main__':
    cfg['data_name'] = 'LIBRISPEECH'

    main()
