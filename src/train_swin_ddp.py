from data import fetch_dataset, make_data_loader
from models.codec import *

from utils import show_and_save, check_exists, makedir_exist_ok, PESQ, save
from config import args

import os, json, wandb, random
from tqdm import tqdm
import numpy as np
import transformers

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from collections import OrderedDict
import datetime
import warnings
warnings.filterwarnings("ignore")


torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    dist.init_process_group('nccl', init_method='env://',
                                    timeout=datetime.timedelta(seconds=1800),
                                    rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def init_model(rank):
    configs = {
        "patch_size":[args.freq_patch,args.time_patch], "swin_depth": args.swin_depth,
        "swin_heads": args.swin_heads, "window_size": args.window_size, "mlp_ratio": args.mlp_ratio,
        "in_freq": args.in_freq, "h_dims": args.swin_h_dims, "max_streams":args.max_streams, 
        "proj": args.proj, "overlap": args.overlap, "num_vqs": args.num_vqs, "codebook_size": args.codebook_size, 
        "cosine_similarity": args.cosine_sim,
        "mel_nfft": args.mel_nfft, "mel_bins": args.mel_bins, 
        "fuse_net": args.fuse_net, "scalable": args.scalable, 
        "spec_augment": args.spec_augment, "win_len": args.win_len, "hop_len": args.hop_len, "sr": args.sr,
        "vis": rank == 0
        }
    if rank == 0: 
        json.dump(configs, open(f"{args.save_dir}/{args.wb_exp_name}/config.json", "w", encoding='utf-8'))
        print(f"Saving into {args.save_dir}/{args.wb_exp_name}")

    model = SwinCrossScaleCodec(**configs)
    return model

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(resumed_checkpoint_pth, model, optimizer, rank):
    loc = f'cuda:{rank}'
    checkpoint = torch.load(resumed_checkpoint_pth, map_location=loc)
    new_state_dict = OrderedDict()
    for key, value in checkpoint['model_state_dict'].items():
        if not key.startswith('module.'):
            new_state_dict['module.' + key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    if rank == 0: 
        print(f"Resuming training from Epoch {start_epoch}")
    return model, optimizer, start_epoch

def train_epoch(model, optimizer, scheduler, data_loader, progress_bar, rank):
    model.train()
    if rank == 0: evaluation = {"loss": [], "recon_loss": [], "vq_loss": [], "mel_loss": []}
    for i, input in enumerate(data_loader):
        # Update Model
        input['audio'], input['feat'] = input['audio'].cuda(rank), input['feat'].cuda(rank)
        optimizer.zero_grad()
        outputs = model(x=input["audio"], x_feat=input["feat"], streams=args.max_streams, train=True)
        outputs['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        optimizer.step()
        scheduler.step()

        torch.distributed.barrier()
        
        # Log Training Process
        if rank == 0:
            progress_bar.update(1)
            for key, val in evaluation.items():
                evaluation[key].append(outputs[key].item())
            # Log Losses
            if (i+1) % args.info_steps == 0:
                for key, val in evaluation.items():
                    evaluation[key] = np.mean(val)
                wandb.log(evaluation)
                for key, val in evaluation.items():
                    evaluation[key] = []
            # Optionally Log Visualizations
            # if (i+1) % int(len(data_loader)*args.plot_interval) == 0:
            #     if not args.scalable:
            #         outputs = model(**dict(x=input["audio"], x_feat=input["feat"], streams=args.max_streams, train=False))
            #         show_and_save(input['audio'][0:1, :], outputs['recon_audio'][0:1, :], 
            #                     save_path=f'{args.save_dir}/runs/{model_tag}/train_epoch{epoch}_batch{i+1}', use_wb=args.use_wb)
            #     else:
            #         outputs = [model(**dict(x=input["audio"], x_feat=input["feat"], streams=j, train=False)) for j in range(1, args.max_streams+1)]
            #         raw_aud, raw_stft, recon_auds, recon_stfts = input['audio'][0, :], input['feat'][0, :], [output['recon_audio'][0, :] for output in outputs], [output['recon_feat'][0, :] for output in outputs]
            #         show_and_save_multiscale(raw_aud, raw_stft, recon_auds, recon_stfts, path=f'{save_path}/runs/{model_tag}/train_epoch{epoch}_batch{i+1}.jpg', mel=True, use_wb=cfg.use_wb)

def validate_epoch(model, data_loader, rank):
    model.eval()
    obj_metric = []
    if rank == 0:
        print("Validating Epoch...")
        progress_bar = tqdm(range(len(data_loader)))
    loc=f"cuda:{rank}"
    with torch.no_grad():
        for i, input in enumerate(data_loader):
            input['audio'], input['feat'] = input['audio'].cuda(rank), input['feat'].cuda(rank)
            outputs = model(x=input["audio"], x_feat=input["feat"], streams=args.max_streams, train=False)
            obj_metric.extend(
                [PESQ(input['audio'][j].cpu().numpy(), outputs['recon_audio'][j].cpu().numpy()) for j in range(input['audio'].size(0))]
            )
            if rank == 0: 
                progress_bar.update(1)
    obj_metric_tensor = torch.tensor(obj_metric, device=loc)
    # print(f"rank: {rank}", obj_metric_tensor)
    gathered_metrics = [torch.zeros_like(obj_metric_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_metrics, obj_metric_tensor)

    if rank == 0:
        # print("gathered_metric: ", gathered_metrics)
        all_metrics = torch.cat(gathered_metrics)
        # print("all_metrics: ", all_metrics)
        test_performance = all_metrics.mean().item()
        wandb.log({"Test_PESQ": test_performance})

    return test_performance if rank == 0 else obj_metric_tensor.mean().item()

def main(rank, world_size):

    setup(rank, world_size)
    if rank == 0:
        if args.wb_project_name is not None:
            wandb.login()
            wandb.init(project=args.wb_project_name, name=args.wb_exp_name)
        else:
            print("Deactivated WandB Logging")

    # Initialize Model Optimizer Scheduler
    model = init_model(rank).cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = transformers.get_constant_schedule(optimizer)

    # Load Dataset
    dataset = fetch_dataset("DNS_CHALLENGE", data_dir=args.data_dir, in_freq=args.in_freq)
    sampler = {"train": DistributedSampler(dataset["train"], num_replicas=world_size, rank=rank), 
               "test": DistributedSampler(dataset["test"], num_replicas=world_size, rank=rank, shuffle=False)}
    data_loaders = make_data_loader(dataset, 
                                    batch_size={"train":args.train_bs_per_device, "test":args.test_bs_per_device}, 
                                    shuffle={"train":False, "test":False}, sampler=sampler)

    train_step_per_epoch, test_step_per_epoch = len(data_loaders['train']), len(data_loaders['test'])
    args.max_train_steps = train_step_per_epoch*args.epochs
    if rank == 0:
        print(f"batch_size_per_device: train {args.train_bs_per_device}, test {args.test_bs_per_device}")
        print("training_steps_per_epoch: {} , total_training_steps: {}".format(train_step_per_epoch, args.max_train_steps))

    # Resume Training optionally
    resumed_checkpoint_pth = f"{args.save_dir}/{args.wb_exp_name}/{args.wb_exp_name}_checkpoint.pt"
    if os.path.exists(resumed_checkpoint_pth):
        model, optimizer, start_epoch = load_checkpoint(resumed_checkpoint_pth, model, optimizer, rank)
        test_perf = [2.529, 2.657]
    else:
        start_epoch = 1
        test_perf = []

    # Training 
    progress_bar = tqdm(initial=(start_epoch-1)*train_step_per_epoch, total=args.max_train_steps, position=0, leave=True)
    for epoch in range(start_epoch, args.epochs+1):
        if rank == 0: print(f"Epoch {epoch} Training:")
        sampler["train"].set_epoch(epoch)
        train_epoch(model, optimizer, scheduler, data_loaders["train"], progress_bar, rank)
        performance = validate_epoch(model, data_loaders["test"], rank)

        if rank == 0:
            print(f"Test Epoch {epoch}, PESQ: {performance}")
            checkpoint = {'epoch': epoch, 
                      'model_state_dict': model.module.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(), 
                      'scheduler_state_dict': scheduler.state_dict()}
            save(checkpoint, '{}/{}/checkpoint.pt'.format(args.save_dir, args.wb_exp_name))

            if epoch > 1:
                if performance >= max(test_perf):
                    print(f"Saving Best Model at Epoch {epoch}")
                    save(checkpoint, '{}/{}/best.pt'.format(args.save_dir, args.wb_exp_name))
            test_perf.append(performance)

    if rank == 0:
        wandb.finish()

    cleanup()

if __name__ == "__main__":
    
    world_size = args.num_device
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)


"""
python train_swin_ddp.py \
    --max_streams 6 \
    --swin_h_dims 45 45 72 96 192 384 \
    --swin_heads 3 3 6 12 24 \
    --proj 8 8 4 4 4 4 \
    --overlap 4 \
    --wb_exp_name swin-9k-scale-baseline \
    --scalable \
    --epochs 60 \
    --lr 1.0e-4 \
    --train_bs_per_device 15 \
    --test_bs_per_device 4 \
    --num_device 4 \
    --seed 830

python train_swin_ddp.py \
    --max_streams 6 \
    --swin_h_dims 45 45 72 96 192 384 \
    --swin_heads 3 3 6 12 24 \
    --proj 16 16 16 16 16 16 \
    --overlap 4 \
    --wb_exp_name swin-9k-scale-vqimprove \
    --cosine_sim \
    --scalable \
    --epochs 60 \
    --lr 1.0e-4 \
    --train_bs_per_device 15 \
    --test_bs_per_device 4 \
    --num_device 4 \
    --seed 830

python train_swin_ddp.py \
    --max_streams 6 \
    --overlap 2 \
    --wb_project_name deep-audio-compress \
    --wb_exp_name swin-18k-scale-shifted-window-attn-fuse \
    --scalable \
    --fuse_net \
    --shift_wa_fuse \
    --epochs 60 \
    --lr 1.0e-4 \
    --train_bs_per_device 15 \
    --test_bs_per_device 4 \
    --num_device 4 \
    --seed 830

python train_swin_ddp.py \
    --max_streams 6 \
    --overlap 2 \
    --wb_project_name deep-audio-compress \
    --wb_exp_name swin-18k-scale-window-attn-fuse \
    --scalable \
    --fuse_net \
    --epochs 60 \
    --lr 1.0e-4 \
    --train_bs_per_device 15 \
    --test_bs_per_device 4 \
    --num_device 4 \
    --seed 830

python train_swin_ddp.py \
    --max_streams 6 \
    --overlap 2 \
    --wb_project_name deep-audio-compress \
    --wb_exp_name swin-18k-scale-flatten-attn-fuse \
    --scalable \
    --fuse_net \
    --epochs 60 \
    --lr 1.0e-4 \
    --train_bs_per_device 15 \
    --test_bs_per_device 4 \
    --num_device 4 \
    --seed 830
"""