import argparse
import datetime
import time
import os
import shutil
import torch
import torch.backends.cudnn as cudnn
from utils import save, to_device, process_dataset, make_optimizer, make_scheduler, collate, \
                resume, check_exists, makedir_exist_ok, plot_spectrogram
from config import cfg, process_args

from data import fetch_dataset, make_data_loader
from models import autoencoder
from metrics.metrics import Metric
from logger import make_logger
import math

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['cuda_device'])

def main():
    for i in range(cfg['num_experiments']):
        bitrate = cfg['fixed_bitstream'] * cfg[cfg['model_name']]['Groups'] * math.log2(cfg[cfg['model_name']]['codebook_size']) * 50
        cfg['model_tag'] = '0_{}_csvqcodec_nonscalable_{}kbps'.format(cfg['data_name'],bitrate*0.001)
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return

def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])

    # load_dataset
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    data_loader = make_data_loader(dataset, cfg['model_name'])
    print(f"train_data:{len(data_loader['train'])} test_data:{len(data_loader['test'])}")

    # load_model
    model = autoencoder.init_model()
    model = model.cuda()

    optimizer = make_optimizer(model, cfg['model_name'])
    scheduler = make_scheduler(optimizer, cfg['model_name'])

    metric = Metric({'train': ['Loss','RECON_Loss','CODEBOOK_Loss','MEL_Loss'], 
                     'test': ['Loss','MEL_Loss','audio_PSNR','PESQ']})

    print("model, optimizer, scheduler, metric loading finish!")

    # resume training 
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            model.load_state_dict(result['model_state_dict'])
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        else:
            logger = make_logger('/scratch/yg172/output/runs/train_{}'.format(cfg['model_tag']))
    else:
        last_epoch = 1
        logger = make_logger('/scratch/yg172/output/runs/train_{}'.format(cfg['model_tag']))

    
    print("Start Training")
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        train_csvqcodec(data_loader['train'], model, optimizer, metric, logger, epoch)
        print('Epoch {} Training Finish! Start to Test'.format(epoch))
        test_csvqcodec(data_loader['test'], model, metric, logger, epoch)

        if cfg[cfg['model_name']]['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
        else:
            scheduler.step()
        model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
        result = {'cfg': cfg, 'epoch': epoch + 1, 'model_state_dict': model_state_dict,
                  'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                  'logger': logger}
        save(result, '/scratch/yg172/output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('/scratch/yg172/output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        '/scratch/yg172/output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def train_csvqcodec(data_loader, model, optimizer, metric, logger, epoch, save_image=cfg['save_image']):
    logger.safe(True) 
    model.train(True)

    start_time = time.time()
    for i, input in enumerate(data_loader):
        input_size = len(input['stft_feat'])
        input = collate(input)
        input = to_device(input, cfg['device'])

        # Train
        optimizer.zero_grad()
        
        output = model(input, Bs=cfg['fixed_bitstream'])
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # Evaluation
        evaluation = metric.evaluate(metric.metric_name['train'], input, output)
        logger.append(evaluation, 'train', n=input_size)
        if i % 5 == 0:
            _time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * _time * len(data_loader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                             'Learning rate: {:.8f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))

        if save_image and i % int((len(data_loader) * cfg['plot_interval']) + 1) == 0:
            root_path = "/scratch/yg172/output/runs/train_{}".format(cfg['model_tag'])
            if not check_exists(root_path): makedir_exist_ok(root_path)
            img_path = os.path.join(root_path, f"train_epoch_{epoch}_batch_{i}.jpg")
            with torch.no_grad():
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                plot_spectrogram(input['stft_feat'][:4],output['recon_feat'][:4],img_path,evaluation,
                                                                Bs=cfg['fixed_bitstream'])
            model.train(True)
    logger.safe(False)
    return

def test_csvqcodec(data_loader, model, metric, logger, epoch, save_image=cfg['save_image']):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        plot_batch = {}
        for i, input in enumerate(data_loader):
            input_size = len(input['stft_feat'])
            input = collate(input)
            input = to_device(input, cfg['device'])

            output = model(input, Bs=cfg['fixed_bitstream'])   #fix the highest bitrate for test inference

            if i == 2: 
                plot_batch['raw_feat'] = input['stft_feat'][:4]
                plot_batch['recon_feat'] = output['recon_feat'][:4]
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}'.format(epoch)]}        
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))

        if save_image:
            root_path = "/scratch/yg172/output/runs/train_{}".format(cfg['model_tag'])
            if not check_exists(root_path): makedir_exist_ok(root_path)
            img_path = os.path.join(root_path, f"test_epoch_{epoch}.jpg")
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)

            plot_spectrogram(plot_batch['raw_feat'],plot_batch['recon_feat'],img_path,evaluation, 
                                                            Bs=cfg['fixed_bitstream'])
    logger.safe(False)
    return


if __name__ == "__main__":
    main()