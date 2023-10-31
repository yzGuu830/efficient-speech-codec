from data import fetch_dataset, make_data_loader
from models.codec import *

from utils import show_and_save, check_exists, makedir_exist_ok, PESQ, save
from config import args

import os
import json
import wandb
from tqdm import tqdm
import numpy as np
import transformers


def main():

    wandb.login()
    wandb.init(project="deep-audio-compress", name=args.wb_exp_name)

    dataset = fetch_dataset("DNS_CHALLENGE", data_dir=args.data_dir, in_freq=args.in_freq)
    data_loaders = make_data_loader(dataset, 
                                   batch_size={"train":args.train_bs, "test":args.test_bs}, 
                                   shuffle={"train":True, "test":False})
    train_dataloader, test_dataloader = data_loaders['train'], data_loaders['test']
    train_step_per_epoch, test_step_per_epoch = len(train_dataloader), len(test_dataloader)
    print(f"DNS_CHALLENGE Dataset Loaded: train {train_step_per_epoch} test {test_step_per_epoch}")

    configs = {
        "patch_size":[args.freq_patch,args.time_patch], "swin_depth": args.swin_depth,
        "swin_heads": args.swin_heads, "window_size": args.window_size, "mlp_ratio": args.mlp_ratio,
        "in_freq": args.in_freq, "h_dims": args.swin_h_dims, "max_streams":args.max_streams, 
        "proj": args.proj, "overlap": args.overlap, "num_vqs": args.num_vqs, "codebook_size": args.codebook_size, 
        "mel_nfft": args.mel_nfft, "mel_bins": args.mel_bins, "fuse_net": args.fuse_net, "scalable": args.scalable, 
        "spec_augment": args.spec_augment, "win_len": args.win_len, "hop_len": args.hop_len, "sr": args.sr
    }
    model = SwinCrossScaleCodec(**configs)
    model_tag = args.wb_exp_name

    if not check_exists(f"{args.save_dir}/runs/{model_tag}"):
        makedir_exist_ok(f"{args.save_dir}/runs/{model_tag}")
    if not check_exists(f"{args.save_dir}/{model_tag}"):
        makedir_exist_ok(f"{args.save_dir}/{model_tag}")
    json.dump(configs, open(f"{args.save_dir}/{model_tag}/config.json", "w", encoding='utf-8'))

    print(f"Saving into {args.save_dir}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = transformers.get_constant_schedule(optimizer)

    # Train
    if args.num_device > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ", ".join([str(i) for i in range(args.num_device)])
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.num_device)])
        print("Utilize {} GPUs for DataParallel Training".format(args.num_device))
    model = model.cuda()
    print("Model Loaded, Start Training...")

    test_perf = []
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch} Training:")
        evaluation = {"loss": [], "recon_loss": [], "vq_loss": [], "mel_loss": []}
        for i, input in tqdm(enumerate(train_dataloader)):
            
            input['audio'], input['feat'] = input['audio'].cuda(), input['feat'].cuda()

            optimizer.zero_grad()
            outputs = model(**dict(x=input["audio"], x_feat=input["feat"], streams=args.max_streams, train=True))
            outputs['loss'].mean().backward() if args.num_device > 1 else outputs['loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

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

        obj_scores = []
        print(f"Epoch {epoch} Testing:")
        for i, input in tqdm(enumerate(test_dataloader)):
            input['audio'], input['feat'] = input['audio'].cuda(), input['feat'].cuda()
            outputs = model(**dict(x=input["audio"], x_feat=input["feat"], streams=args.max_streams, train=False))
            obj_scores.extend(
                [PESQ(input['audio'][j].cpu().numpy(), outputs['recon_audio'][j].cpu().numpy()) for j in range(input['audio'].size(0))]
            )
        performance = np.mean(obj_scores)
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

    wandb.finish()
    print("Copy Best Model into folder")
    os.system(f"cp {args.save_dir}/model/{model_tag}_best.pt {args.save_dir}/{model_tag}/best.pt")

if __name__ == "__main__":
    main()

"""
python train_swin.py \
    --max_streams 6 \
    --proj 2 \
    --overlap 2 \
    --use_wb \
    --wb_exp_name swin-18k \
    --epochs 30 \
    --lr 3.0e-4 \
    --train_bs 32 \
    --test_bs 16 \
    --num_device 4 \
    --plot_interval .53

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
    --wb_exp_name swin-18k-scale-tfloss \
    --scalable \
    --epochs 60 \
    --lr 1.0e-4 \
    --train_bs 60 \
    --test_bs 20 \
    --num_device 4 \
    --seed 53 \
    --plot_interval .66
"""