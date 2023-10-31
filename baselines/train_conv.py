from data import fetch_dataset, make_data_loader
from models.codec import *

from utils import show_and_save, check_exists, makedir_exist_ok, PESQ, save
from config import args

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Process

import os
import json
import wandb
from tqdm import tqdm
import numpy as np
import transformers



def main(rank, world_size):
    if world_size > 1:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

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
        "in_freq": args.in_freq, "h_dims": args.h_dims, "max_streams":args.max_streams, 
        "proj": args.proj, "overlap": args.overlap, "num_vqs": args.num_vqs, "codebook_size": args.codebook_size, 
        "mel_nfft": args.mel_nfft, "mel_bins": args.mel_bins, "fuse_net": args.fuse_net, "scalable": args.scalable, 
        "spec_augment": args.spec_augment, "win_len": args.win_len, "hop_len": args.hop_len, "sr": args.sr,
        "use_tf": args.use_tf
    }
    model = ConvCrossScaleCodec(**configs)
    model_tag = args.wb_exp_name

    model = model.cuda()
    # Train
    if args.num_device > 1:
        # os.environ['CUDA_VISIBLE_DEVICES'] = ", ".join([str(i) for i in range(args.num_device)])
        # model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.num_device)])
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        if rank==0: print("Utilize {} GPUs for DistributedDataParallel Training".format(args.num_device))
    
    if not check_exists(f"{args.save_dir}/runs/{model_tag}"):
        makedir_exist_ok(f"{args.save_dir}/runs/{model_tag}")
    if not check_exists(f"{args.save_dir}/{model_tag}"):
        makedir_exist_ok(f"{args.save_dir}/{model_tag}")
    json.dump(configs, open(f"{args.save_dir}/{model_tag}/config.json", "w", encoding='utf-8'))
    if rank==0: print(f"Saving into {args.save_dir}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = transformers.get_constant_schedule(optimizer)
    
    if rank==0: print("Model Loaded, Start Training...")

    test_perf = []
    for epoch in range(1, args.epochs + 1):
        if world_size>1: sampler["train"].set_epoch(epoch)
        if rank==0: print(f"Epoch {epoch} Training:")
        evaluation = {"loss": [], "recon_loss": [], "vq_loss": [], "mel_loss": []}
        for i, input in tqdm(enumerate(train_dataloader)):
    
            input['audio'], input['feat'] = input['audio'].cuda(), input['feat'].cuda()

            optimizer.zero_grad()
            outputs = model(**dict(x=input["audio"], x_feat=input["feat"], streams=args.max_streams, train=True))
            outputs['loss'].mean().backward() if args.num_device > 1 else outputs['loss'].backward()

            optimizer.step()
            scheduler.step()

            loss_val, recon_loss_val, vq_loss_val, mel_loss_val = outputs['loss'].mean().item() if args.num_device > 1 else outputs['loss'].item(), \
                    outputs['recon_loss'].mean().item() if args.num_device > 1 else outputs['recon_loss'].item(), \
                    outputs['vq_loss'].mean().item() if args.num_device > 1 else outputs['vq_loss'].item(), \
                    outputs['mel_loss'].mean().item() if args.num_device > 1 else outputs['mel_loss'].item()
            evaluation['loss'].append(loss_val)
            evaluation['recon_loss'].append(recon_loss_val)
            evaluation['vq_loss'].append(vq_loss_val)
            evaluation['mel_loss'].append(mel_loss_val)

            if rank==0:
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
                # else:
                #     outputs = [model(**dict(x=input["audio"], x_feat=input["feat"], streams=j, train=False)) for j in range(1, args.max_streams+1)]
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

        if rank == 0:
            wandb.log({"Test_PESQ": performance})
            print(f"Test Epoch {epoch} || PESQ: {performance}")

            if not args.scalable:
                outputs = model(**dict(x=input["audio"], x_feat=input["feat"], streams=args.max_streams, train=False))
                if not check_exists(f'{args.save_dir}/runs/{model_tag}/test_epoch{epoch}'):
                    makedir_exist_ok(f'{args.save_dir}/runs/{model_tag}/test_epoch{epoch}')
                show_and_save(input['audio'][0:1, :], outputs['recon_audio'][0:1, :], 
                                save_path=f'{args.save_dir}/runs/{model_tag}/test_epoch{epoch}', use_wb=False)
        # else:
        #     outputs = [model(**dict(x=input["audio"], x_feat=input["feat"], streams=j, train=False)) for j in range(1, args.max_streams+1)]
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
    
if __name__ == "__main__":
    wandb.login()
    wandb.init(project="csvq-reproduce", name=args.wb_exp_name)

    world_size = args.num_device
    if world_size == 1:
        main(rank=0, world_size=world_size)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        processes = []
        for rank in range(world_size):
            p = Process(target=main, args=(rank, world_size))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    if world_size > 1:
        dist.destroy_process_group()
    wandb.finish()
    print("Copy Best Model into folder")
    os.system(f"cp {args.save_dir}/model/{args.wb_exp_name}_best.pt {args.save_dir}/{args.wb_exp_name}/best.pt")

"""
python train_conv.py \
    --max_streams 6 \
    --use_wb \
    --wb_exp_name csvq-tcm-merge18k-nonscale-baseline \
    --fuse_net \
    --use_tf \
    --epochs 50 \
    --lr 3.0e-4 \
    --train_bs_per_device 32 \
    --test_bs_per_device 8 \
    --proj 2 \
    --overlap 4 \
    --num_device 1 \
    --plot_interval .53

python train_conv.py \
    --max_streams 6 \
    --use_wb \
    --wb_exp_name csvq-tcm-merge18k-scale-baseline \
    --scalable \
    --fuse_net \
    --use_tf \
    --epochs 50 \
    --lr 3.0e-4 \
    --train_bs_per_device 32 \
    --test_bs_per_device 8 \
    --proj 2 \
    --overlap 4 \
    --num_device 1 \
    --plot_interval .53

python train_conv.py \
    --max_streams 6 \
    --use_wb \
    --wb_exp_name csvq-notcm-merge18k-scale-baseline \
    --scalable \
    --fuse_net \
    --epochs 50 \
    --lr 3.0e-4 \
    --train_bs_per_device 32 \
    --test_bs_per_device 8 \
    --proj 2 \
    --overlap 4 \
    --num_device 1 \
    --plot_interval .53

python train_conv.py \
    --max_streams 6 \
    --use_wb \
    --wb_exp_name csvq-tcm-residual18k-scale-baseline \
    --scalable \
    --use_tf \
    --epochs 50 \
    --lr 3.0e-4 \
    --train_bs_per_device 32 \
    --test_bs_per_device 8 \
    --proj 2 \
    --overlap 4 \
    --num_device 1 \
    --plot_interval .53

python train_conv.py \
    --max_streams 6 \
    --use_wb \
    --wb_exp_name csvq-notcm-residual18k-scale-baseline \
    --scalable \
    --epochs 50 \
    --lr 3.0e-4 \
    --train_bs_per_device 32 \
    --test_bs_per_device 8 \
    --proj 2 \
    --overlap 4 \
    --num_device 1 \
    --plot_interval .53

python train_conv.py \
    --max_streams 6 \
    --use_wb \
    --wb_exp_name csvq-residual18k-scale-baseline \
    --scalable \
    --epochs 50 \
    --lr 3.0e-4 \
    --train_bs_per_device 32 \
    --test_bs_per_device 8 \
    --proj 2 \
    --overlap 4 \
    --num_device 1 \
    --plot_interval .53
"""