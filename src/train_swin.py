from data import fetch_dataset, make_data_loader
from models.codec import *

from utils import show_and_save, check_exists, makedir_exist_ok, PESQ, save
from config import args

import os
import json
import wandb
from tqdm import tqdm
import numpy as np
from datetime import timedelta
import transformers

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo',
                            rank=rank,
                            world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    # Destroy the process group
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    if rank==0:
        wandb.login()
        wandb.init(project="deep-audio-compress", name=args.wb_exp_name)

    dataset = fetch_dataset("DNS_CHALLENGE", data_dir=args.data_dir, in_freq=args.in_freq)
    sampler = {"train": DistributedSampler(dataset["train"]),
                "test": DistributedSampler(dataset["test"])} if args.num_device > 1 else None
    data_loaders = make_data_loader(dataset, 
                                   batch_size={"train":args.train_bs_per_device, "test":args.test_bs_per_device}, 
                                   shuffle={"train":True, "test":False},
                                   sampler=sampler)
    train_dataloader, test_dataloader = data_loaders['train'], data_loaders['test']
    train_step_per_epoch, test_step_per_epoch = len(train_dataloader), len(test_dataloader)
    if rank==0: print(f"DNS_CHALLENGE Dataset Loaded: train {train_step_per_epoch} test {test_step_per_epoch}")

    configs = {
        "patch_size":[args.freq_patch,args.time_patch], "swin_depth": args.swin_depth,
        "swin_heads": args.swin_heads, "window_size": args.window_size, "mlp_ratio": args.mlp_ratio,
        "in_freq": args.in_freq, "h_dims": args.swin_h_dims, "max_streams":args.max_streams, 
        "proj": args.proj, "overlap": args.overlap, "num_vqs": args.num_vqs, "codebook_size": args.codebook_size, 
        "mel_nfft": args.mel_nfft, "mel_bins": args.mel_bins, "fuse_net": args.fuse_net, "scalable": args.scalable, 
        "spec_augment": args.spec_augment, "win_len": args.win_len, "hop_len": args.hop_len, "sr": args.sr
    }
    model = SwinCrossScaleCodec(**configs).to(rank)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = transformers.get_constant_schedule(optimizer)

    # Load the state dict properly if model was saved using DataParallel
    if os.path.exists(f"{args.save_dir}/{args.wb_exp_name}/{args.wb_exp_name}_checkpoint.pt"):
        if rank==0: print("Resume Training")
        checkpoint = torch.load(f"{args.save_dir}/{args.wb_exp_name}/{args.wb_exp_name}_checkpoint.pt", map_location="cuda:{}".format(rank))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if rank==0: print(f"Resuming training from epoch {start_epoch}")
        test_perf = [2.529, 2.657]
    else:
        start_epoch = 1
        test_perf = []

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if rank==0:
        model_tag = args.wb_exp_name
        if not check_exists(f"{args.save_dir}/runs/{model_tag}"):
            makedir_exist_ok(f"{args.save_dir}/runs/{model_tag}")
        if not check_exists(f"{args.save_dir}/{model_tag}"):
            makedir_exist_ok(f"{args.save_dir}/{model_tag}")
        json.dump(configs, open(f"{args.save_dir}/{model_tag}/config.json", "w", encoding='utf-8'))
        print(f"Saving into {args.save_dir}")

    # # Train
    # if args.num_device > 1:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = ", ".join([str(i) for i in range(args.num_device)])
    #     model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.num_device)])
    #     print("Utilize {} GPUs for DataParallel Training".format(args.num_device))
    # model = model.to("cuda")
    # print("Model Loaded, Start Training...")

    for epoch in range(start_epoch, args.epochs + 1):
        if rank==0: print(f"Epoch {epoch} Training:")
        if world_size>1: sampler["train"].set_epoch(epoch)
        evaluation = {"loss": [], "recon_loss": [], "vq_loss": [], "mel_loss": []}
        for i, input in tqdm(enumerate(train_dataloader)):
            
            input['audio'], input['feat'] = input['audio'].cuda(), input['feat'].cuda()

            optimizer.zero_grad()
            outputs = model(**dict(x=input["audio"], x_feat=input["feat"], streams=args.max_streams, train=True))
            outputs['loss'].mean().backward() if args.num_device > 1 else outputs['loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

            optimizer.step()
            scheduler.step()

            if rank==0:
                loss_val, recon_loss_val, vq_loss_val, mel_loss_val = outputs['loss'].mean().item() if args.num_device > 1 else outputs['loss'].item(), \
                        outputs['recon_loss'].mean().item() if args.num_device > 1 else outputs['recon_loss'].item(), \
                        outputs['vq_loss'].mean().item() if args.num_device > 1 else outputs['vq_loss'].item(), \
                        outputs['mel_loss'].mean().item() if args.num_device > 1 else outputs['mel_loss'].item()
                evaluation['loss'].append(loss_val)
                evaluation['recon_loss'].append(recon_loss_val)
                evaluation['vq_loss'].append(vq_loss_val)
                evaluation['mel_loss'].append(mel_loss_val)

                if (i+1) % args.info_steps == 0:
                    wandb_log = {}
                    for key, val in evaluation.items():
                        wandb_log[key] = np.mean(val)
                    wandb.log(wandb_log)
                    for key, val in evaluation.items():
                        evaluation[key].clear()

                if (i+1) % int(train_step_per_epoch*args.plot_interval) == 0:
                    if not args.scalable:
                        outputs = model(**dict(x=input["audio"], x_feat=input["feat"], streams=args.max_streams, train=False))
                        show_and_save(input['audio'][0:1, :], outputs['recon_audio'][0:1, :], 
                                    save_path=f'{args.save_dir}/runs/{model_tag}/train_epoch{epoch}_batch{i+1}', use_wb=args.use_wb)
                    else:
                        outputs = [model(**dict(x=input["audio"], x_feat=input["feat"], streams=j, train=False)) for j in range(1, args.max_streams+1)]
                        # raw_aud, raw_stft, recon_auds, recon_stfts = input['audio'][0, :], input['feat'][0, :], [output['recon_audio'][0, :] for output in outputs], [output['recon_feat'][0, :] for output in outputs]
                        # show_and_save_multiscale(raw_aud, raw_stft, recon_auds, recon_stfts, path=f'{save_path}/runs/{model_tag}/train_epoch{epoch}_batch{i+1}.jpg', mel=True, use_wb=cfg.use_wb)
        
        print(f"Epoch {epoch} Rank {rank} Testing:")
        local_obj_scores = []
        for i, input in tqdm(enumerate(test_dataloader)):
            input['audio'], input['feat'] = input['audio'].cuda(), input['feat'].cuda()
            outputs = model(**dict(x=input["audio"], x_feat=input["feat"], streams=args.max_streams, train=False))
            local_obj_scores.extend(
                [PESQ(input['audio'][j].cpu().numpy(), outputs['recon_audio'][j].cpu().numpy()) for j in range(input['audio'].size(0))]
            )
        if world_size > 1: #DDP
            local_obj_tensor = torch.tensor(local_obj_scores).cuda()
            global_obj_scores = [torch.zeros_like(local_obj_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(global_obj_scores, local_obj_tensor)
            global_obj_scores = torch.cat(global_obj_scores).cpu().numpy()
            performance = np.mean(global_obj_scores)
        else: 
            performance = np.mean(local_obj_scores)
        
        if rank==0:
            wandb.log({"Test_PESQ": performance})
            print(f"Test Epoch {epoch} || PESQ: {performance}")
            if not args.scalable:
                outputs = model(**dict(x=input["audio"], x_feat=input["feat"], streams=args.max_streams, train=False))
                if not check_exists(f'{args.save_dir}/runs/{model_tag}/test_epoch{epoch}'):
                    makedir_exist_ok(f'{args.save_dir}/runs/{model_tag}/test_epoch{epoch}')
                show_and_save(input['audio'][0:1, :], outputs['recon_audio'][0:1, :], 
                                save_path=f'{args.save_dir}/runs/{model_tag}/test_epoch{epoch}', use_wb=False)
            else:
                outputs = [model(**dict(x=input["audio"], x_feat=input["feat"], streams=j, train=False)) for j in range(1, args.max_streams+1)]
                # raw_aud, raw_stft, recon_auds, recon_stfts = input['audio'][0, :], input['feat'][0, :], [output['recon_audio'][0, :] for output in outputs], [output['recon_feat'][0, :] for output in outputs]
                # show_and_save_multiscale(raw_aud, raw_stft, recon_auds, recon_stfts, path=f'{save_path}/runs/{model_tag}/test_epoch{epoch}.jpg', mel=True, use_wb=False)

            model_state_dict = model.module.state_dict() if args.num_device > 1 else model.state_dict()
            result = {'epoch': epoch, 'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scheduler_state_dict': scheduler.state_dict()}
            save(result, '{}/model/{}_checkpoint.pt'.format(args.save_dir, model_tag))

            if epoch > 1:
                if performance >= max(test_perf):
                    print(f"Saved Best Model at Epoch {epoch}")
                    save(result, '{}/model/{}_best.pt'.format(args.save_dir, model_tag))
            test_perf.append(performance)
    if rank==0:
        wandb.finish()
        print("Copy Best Model into folder")
        os.system(f"cp {args.save_dir}/model/{args.wb_exp_name}_best.pt {args.save_dir}/{args.wb_exp_name}/best.pt")
    cleanup()

if __name__ == "__main__":
    
    world_size = args.num_device
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

"""

python train_swin.py \
    --max_streams 6 \
    --proj 4 \
    --overlap 2 \
    --swin_heads 3 \
    --use_wb \
    --wb_exp_name swin-18k-nonscale-baseline \
    --epochs 40 \
    --lr 3.0e-4 \
    --train_bs 60 \
    --test_bs 24 \
    --num_device 4 \
    --seed 830 \
    --plot_interval .53

python train_swin.py \
    --max_streams 6 \
    --overlap 2 \
    --use_wb \
    --wb_exp_name swin-18k-scale-baseline \
    --scalable \
    --epochs 60 \
    --lr 1.0e-4 \
    --train_bs 60 \
    --test_bs 20 \
    --num_device 4 \
    --seed 53 \
    --plot_interval .66

python train_swin.py \
    --max_streams 6 \
    --overlap 2 \
    --use_wb \
    --wb_exp_name swin-18k-scale-attn-fuse \
    --scalable \
    --fuse_net \
    --epochs 60 \
    --lr 1.0e-4 \
    --train_bs 30 \
    --test_bs 12 \
    --num_device 4 \
    --seed 53 \
    --plot_interval .66


python train_swin.py \
    --max_streams 6 \
    --overlap 2 \
    --use_wb \
    --wb_exp_name swin-18k-scale-flatten-attn-fuse \
    --scalable \
    --fuse_net \
    --epochs 60 \
    --lr 1.0e-4 \
    --train_bs_per_device 15 \
    --test_bs_per_device 4 \
    --num_device 4 \
    --seed 53 \
    --plot_interval .66
"""