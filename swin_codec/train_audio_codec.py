import os
from tqdm import tqdm

from data import fetch_dataset, make_data_loader

from models.autoencoder import Swin_Audio_Codec
import torch
import transformers
from utils import save, show_and_save, make_obj_score, show_and_save_multiscale
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
import json

def main():
    train_bs, test_bs = cfg.train_bs, cfg.test_bs
    num_workers = cfg.num_workers
    warmup_epochs, epochs = cfg.warmup_epochs, cfg.epochs
    plot_interval = cfg.plot_interval
    data_name = cfg.data_name
    num_streams = cfg.num_streams

    # Load Dataset 
    dataset = fetch_dataset(data_name)
    data_loader = make_data_loader(dataset, {"train":train_bs, "test":test_bs}, {"train":True, "test":False})
    train_dataloader, test_dataloader = data_loader['train'], data_loader['test']

    train_batch, test_batch = len(train_dataloader), len(test_dataloader)
    print(f"{data_name} Dataset Loaded: train {train_batch} test {test_batch}")

    # Load Model 
    config = {
        "init_H":cfg.init_H, "in_channels":cfg.in_channels, "patch_size":cfg.patch_size, 
        "model_depth":cfg.model_depth, "layer_depth":cfg.layer_depth,
        "d_model":cfg.d_model, "num_heads":cfg.num_heads, "window_size":cfg.window_size, 
        "mlp_ratio":cfg.mlp_ratio, "qkv_bias":cfg.qkv_bias, "qk_scale":cfg.qk_scale, 
        "proj_drop":cfg.proj_drop, "attn_drop":cfg.attn_drop, "norm_layer":cfg.norm_layer,
        "vq_down_ratio":cfg.vq_down_ratio, "num_overlaps":cfg.num_overlaps, 
        "num_groups":cfg.num_groups, "codebook_size":cfg.codebook_size, "vq_commit":cfg.vq_commit,
        "win_length":cfg.win_length, "hop_length":cfg.hop_length, "sr":cfg.sr, "scalable":cfg.scalable
    }
    Model = Swin_Audio_Codec(**config)

    model_tag = f"{data_name}_Audio_Codec_{Model.max_bps}kbps_scalable_PatchSize{cfg.patch_size[0]}_{cfg.patch_size[1]}" if cfg.scalable \
        else f"{data_name}_Audio_Codec_{Model.max_bps}kbps_PatchSize{cfg.patch_size[0]}_{cfg.patch_size[1]}"
    
    if cfg.pretrain:
        print("Pretraining Stage")
        num_streams = 0
        model_tag = f"{data_name}_Audio_Codec_15.0kbps_PatchSize{cfg.patch_size[0]}_{cfg.patch_size[1]}_Pretrain"
        warmup_epochs, epochs = 1, 10

    save_path = cfg.save_path
    print(f"Experiment: {model_tag}")

    if not os.path.exists(f"{save_path}/runs/{model_tag}"): 
        os.makedirs(f"{save_path}/runs/{model_tag}")

    if not os.path.exists(f"{save_path}/{model_tag}"): 
        os.makedirs(f"{save_path}/{model_tag}")
    with open(f"{save_path}/{model_tag}/config.json", "w", encoding='utf-8') as j:
        json.dump(config, j)
        j.close()
    
    optimizer = torch.optim.Adam(Model.parameters(), lr=cfg.lr)

    scheduler = transformers.get_cosine_schedule_with_warmup(
                            optimizer, 
                            num_warmup_steps=warmup_epochs*train_batch,
                            num_training_steps=epochs*train_batch) if cfg.cosine_warmup \
                    else transformers.get_constant_schedule(optimizer)
    print(f"Warmup Epochs: {warmup_epochs}  Epochs: {epochs}  Max Learning Rate: {cfg.lr}")

    ## Training 
    if num_workers > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"
        Model = torch.nn.DataParallel(Model, device_ids = [0,1,2,3])
        print("Utilize {} GPUs for Parallel Training".format(num_workers))

    Model = Model.cuda()
    print("Model Loaded, Start Training...")

    learning_curve, test_perf = {'train':[], 'test':[]}, []
    for epoch in range(1, epochs + 1):

        Model.train()
        evaluation = {"loss": [], "recon_loss": [], "vq_loss": [], "mel_loss": [], "ms_loss": []}
        for i, input in enumerate(train_dataloader):
            
            input['audio'], input['feat'] = input['audio'].cuda(), input['feat'].cuda()

            optimizer.zero_grad()
    
            output = Model(**dict(input=input, audio_len=3, num_stream=num_streams, train=True))

            output['loss'].mean().backward() if num_workers > 1 else output['loss'].backward()

            if cfg.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(Model.parameters(), cfg.clip_max_norm)

            optimizer.step()
            scheduler.step()

            loss_val, recon_loss_val, vq_loss_val, mel_loss_val, ms_loss_val = output['loss'].mean().item() if num_workers > 1 else output['loss'].item(), \
                    output['recon_loss'].mean().item() if num_workers > 1 else output['recon_loss'].item(), \
                    output['vq_loss'].mean().item() if num_workers > 1 else output['vq_loss'].item(), \
                    output['mel_loss'].mean().item() if num_workers > 1 else output['mel_loss'].item(), \
                    output['ms_loss'].mean().item() if num_workers > 1 else output['ms_loss'].item()

            evaluation['loss'].append(loss_val)
            evaluation['recon_loss'].append(recon_loss_val)
            evaluation['vq_loss'].append(vq_loss_val)
            evaluation['mel_loss'].append(mel_loss_val)
            evaluation['ms_loss'].append(ms_loss_val)

            if (i+1) % 3 == 0:
                print("Train Epoch {} Batch {}/{} || loss: {:.5f} | recon_loss: {:.5f} | vq_loss: {:.5f} | mel_loss: {:.5f} | ms_loss: {:.5f} | learning rate: {:.8f}".format(
                    epoch, i+1, train_batch, 
                    np.mean(evaluation['loss']), 
                    np.mean(evaluation['recon_loss']), 
                    np.mean(evaluation['vq_loss']),
                    np.mean(evaluation['mel_loss']), 
                    np.mean(evaluation['ms_loss']),
                    optimizer.param_groups[0]['lr']
                ))
                learning_curve['train'].append(np.mean(evaluation['loss']))
                evaluation = {"loss": [], "recon_loss": [], "vq_loss": [], "mel_loss": [], "ms_loss": []}

            if (i+1) % int(train_batch*plot_interval) == 0:
                print(f"Visualize and Test at Epoch {epoch} Batch {i+1}")
                Model.eval()
                with torch.no_grad():
                    if not cfg.scalable:
                        output = Model(**dict(input=input, audio_len=3, num_stream=num_streams, train=False))
                        raw_aud, raw_stft, recon_aud, recon_stft = input['audio'][0, :], input['feat'][0, :], output['recon_audio'][0, :], output['recon_feat'][0, :]
                        show_and_save(raw_aud, raw_stft, recon_aud, recon_stft, path=f'{save_path}/runs/{model_tag}/train_epoch{epoch}_batch{i+1}.jpg')

                    else:
                        outputs = [ Model(**dict(input=input, audio_len=3, num_stream=j, train=False)) for j in range(1, num_streams+1) ]
                        raw_aud, raw_stft, recon_auds, recon_stfts = input['audio'][0, :], input['feat'][0, :], [output['recon_audio'][0, :] for output in outputs], [output['recon_feat'][0, :] for output in outputs]
                        show_and_save_multiscale(raw_aud, raw_stft, recon_auds, recon_stfts, path=f'{save_path}/runs/{model_tag}/train_epoch{epoch}_batch{i+1}.jpg')

                Model.train()

        obj_scores = []
        Model.eval()
        with torch.no_grad():
            for i, input in tqdm(enumerate(test_dataloader)):
                input['audio'], input['feat'] = input['audio'].cuda(), input['feat'].cuda()

                output = Model(**dict(input=input, audio_len=10, num_stream=num_streams, train=False))

                obj_scores.append(
                        np.mean(
                            [make_obj_score(input['audio'][j].cpu().numpy(), output['recon_audio'][j].cpu().numpy()) for j in range(input['audio'].size(0))]
                        )
                )
            performance = np.mean(obj_scores)
            learning_curve['test'].append(performance)
            print(f"Test Epoch {epoch} || PESQ: {performance}")

            if not cfg.scalable:
                output = Model(**dict(input=input, audio_len=10, num_stream=num_streams, train=False))
                raw_aud, raw_stft, recon_aud, recon_stft = input['audio'][0, :], input['feat'][0, :], output['recon_audio'][0, :], output['recon_feat'][0, :]
                show_and_save(raw_aud, raw_stft, recon_aud, recon_stft, path=f'{save_path}/runs/{model_tag}/test_epoch{epoch}.jpg')

            else:
                outputs = [ Model(**dict(input=input, audio_len=10, num_stream=j, train=False)) for j in range(1, num_streams+1) ]
                raw_aud, raw_stft, recon_auds, recon_stfts = input['audio'][0, :], input['feat'][0, :], [output['recon_audio'][0, :] for output in outputs], [output['recon_feat'][0, :] for output in outputs]
                show_and_save_multiscale(raw_aud, raw_stft, recon_auds, recon_stfts, path=f'{save_path}/runs/{model_tag}/test_epoch{epoch}.jpg')

        model_state_dict = Model.module.state_dict() if num_workers > 1 else Model.state_dict()
        result = {'epoch': epoch, 'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(), 
                'scheduler_state_dict': scheduler.state_dict()}
        save(result, '{}/model/{}_checkpoint.pt'.format(save_path, model_tag))

        if epoch > 1:
            if performance >= max(test_perf):
                print(f"Saved Best Model at Epoch {epoch}")
                save(result, '{}/model/{}_best.pt'.format(save_path, model_tag))
        
        test_perf.append(performance)

        fig, axs = plt.subplots(ncols=2, figsize=(12,3.5))
        axs[0].plot(learning_curve['train'][50:])
        axs[0].set_title("Training Loss")

        axs[1].plot(learning_curve['test'])
        axs[1].set_title("Testing PESQ")

        fig.savefig(f'{save_path}/runs/{model_tag}/learning_curve.jpg')
    
    print("Copy Best Model into folder")
    os.system(f"cp {save_path}/model/{model_tag}_best.pt {save_path}/{model_tag}/best.pt")

if __name__ == "__main__":
    main()
    """
    python train_audio_codec.py \
        --scalable \
        --data_name "DNS_CHALLENGE" \
        --num_streams 6 \
        --patch_size (3,2) \
        --lr 1.0e-4 \
        --train_bs 72 \
        --test_bs 36 \
        --epochs 50 \
        --plot_interval .66

    """