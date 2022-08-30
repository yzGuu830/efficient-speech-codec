import argparse
import copy
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, collate ,resume
from logger import make_logger
from config import cfg, process_args
from models import vqvae

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)
torch.cuda.empty_cache()

def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
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
    print("data_loader loading finish!")
    # load_model
    print("loading model, optimizer, scheduler, metric...")
    model = eval('vqvae.vqvae().to(cfg["device"])')
    optimizer = make_optimizer(model, cfg['model_name'])
    scheduler = make_scheduler(optimizer, cfg['model_name'])
    metric = Metric({'train': ['Loss','Perplexity'], 'test': ['Loss', 'Perplexity', 'PESQ']})
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

    # parallel training
    # if cfg['world_size'] > 1:
    #     model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))
    

    print("Start Training")
    # training epoch
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        
        train(data_loader['train'], model, optimizer, metric, logger, epoch)
        # test_model = make_batchnorm_stats(dataset['train'], model, cfg['model_name'])
        torch.cuda.empty_cache()
        test(data_loader['test'], model, metric, logger, epoch)
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


def train(data_loader, model, optimizer, metric, logger, epoch):
    logger.safe(True)
    model.train(True)
    
    start_time = time.time()
    
    for i, input in enumerate(data_loader):
        
        input = collate(input) # input: {['data':[]]...} -> {'data':[]}
    
        input_size = input['data'].size(0)
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        output = model(input['data'])
        
        output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        
        evaluation = metric.evaluate(metric.metric_name['train'], input, output)

        logger.append(evaluation, 'train', n=input_size)
        # if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
        if i % 10 == 0:
            _time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * _time * len(data_loader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input['data'])
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


if __name__ == "__main__":
    # from utils import process_dataset, process_control, collate
    # process_control()
    # cfg['seed'] = 0

    # # load_dataset
    # dataset = fetch_dataset(cfg['data_name'])
    # process_dataset(dataset)
    # data_loader = make_data_loader(dataset, cfg['model_name'])


    # from models import vqvae

    # model = vqvae.VQVAE(h_dim=128, res_h_dim=32, n_res_layers=2, 
    #              n_embeddings=512, embedding_dim=64, beta=.25)
    
    # # load_model
    # # model = eval('models.{}.{}().to(cfg["device"])'.format(cfg['model_name'],cfg['model_name']))
    # optimizer = make_optimizer(model, cfg['model_name'])
    # scheduler = make_scheduler(optimizer, cfg['model_name'])
    # metric = Metric({'train': ['Loss', 'PSNR', 'PESQ'], 'test': ['Loss', 'PSNR', 'PESQ']})


    # # train
    main()
    
    
            
