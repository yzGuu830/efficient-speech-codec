from models.codec import SwinAudioCodec
from utils import manage_checkpoint, dict2namespace
from data import fetch_dataset, make_data_loader
from models.losses.metrics import PESQ, MelDistance, SISDRLoss

import argparse
import torch, json, os, yaml
import numpy as np
from tqdm import tqdm

def init_args_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_pth", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--data_pth", type=str)

    parser.add_argument("--bit_per_stream", type=float, default=3.0)
    parser.add_argument("--num_worker", default=0, type=int)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    with open(os.path.join('./configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    return args, config

class SwinAudioCodecEval(SwinAudioCodec):
    def __init__(self, config) -> None:
        super().__init__(**vars(config.model))
        pass

    def from_pretrain(self, path):
        
        ckp = torch.load(f"{path}/best.pt",map_location="cpu")
        new_state_dict = manage_checkpoint(ckp)
        self.load_state_dict(new_state_dict)
        
        print(f"Pretrained Model {path.split('/')[-1]} Loaded")

def eval_multi_scale(args, config):

    # Model
    codec = SwinAudioCodecEval(config)
    codec.from_pretrain(args.weight_pth)
    codec = codec.to(args.device)

    # Metrics
    mel_distance_metric = MelDistance(
            win_lengths=[32,64,128,256,512,1024,2048],
            n_mels=[5,10,20,40,80,160,320],
            clamp_eps=1e-5
        ).to(args.device)
    sisdr_metric = SISDRLoss(
        scaling=True, reduction="none",
        zero_mean=True, clip_min=None, weight=1.0,
    )
    pesq_metric = PESQ(sample_rate=16000, device=args.device)

    # Data
    datasets = fetch_dataset("DNS_CHALLENGE", data_dir=config.data.data_dir, in_freq=config.data.in_freq,
                        win_len=config.data.win_len, hop_len=config.data.hop_len, sr=config.data.sr)
    dls = make_data_loader(datasets, 
                    batch_size={"train": 20, "test": 16}, 
                    shuffle={"train": False, "test": False}, 
                    sampler={"train": None, "test": None}, 
                    num_workers=args.num_worker)
    test_dl = dls["test"]

    # Evaluate loop
    performance_table = {}
    for s in range(1, 7):
        test_perf = {"mel_dist": [], "si-sdr": [], "pesq": []}
        for i, input in tqdm(enumerate(test_dl), total=len(test_dl), desc=f"Eval at {s*args.bit_per_stream:.2f}kbps"):
            input['audio'], input['feat'] = input['audio'].to(args.device), None
            outputs = codec(**dict(x=input["audio"], x_feat=input["feat"], streams=s, train=False))

            test_perf["mel_dist"].extend(mel_distance_metric(input["audio"], outputs['recon_audio']).tolist())
            test_perf["si-sdr"].extend(sisdr_metric(input["audio"], outputs['recon_audio']).tolist())
            test_perf["pesq"].extend(pesq_metric(input["audio"], outputs['recon_audio']))

        for k, v in test_perf.items():
            test_perf[k] = np.mean(v)
        performance_table[f"{s*args.bit_per_stream:.2f}kbps"] = test_perf
        print(f"{s*args.bit_per_stream:.2f}kbps: {test_perf}")
    print("Saving Full Performance Table into ", f"{args.weight_pth}/performance.json")
    json.dump(performance_table, open(f"{args.weight_pth}/performance.json", 'w'), indent=4)

if __name__ == "__main__":
    args, config = init_args_configs()
    eval_multi_scale(args, config)

    
    
"""
export CUDA_VISIBLE_DEVICES=0
python evaluate.py \
    --config residual_18k.yml \
    --weight_pth ../output/swin-18k-residual \
    --data_pth ../data/DNS_CHALLENGE/processed_yz \
    --bit_per_stream 3.0 \
    --device cuda 

python evaluate.py \
    --config residual_18k_vq_control.yml \
    --weight_pth ../output/swin-18k-residual-vq-control \
    --data_pth ../data/DNS_CHALLENGE/processed_yz \
    --bit_per_stream 3.0 \
    --device cuda 

python evaluate.py \
    --config residual_18k_vq_ema.yml \
    --weight_pth ../output/swin-18k-residual-vq-ema \
    --data_pth ../data/DNS_CHALLENGE/processed_yz \
    --bit_per_stream 3.0 \
    --device cuda 

python evaluate.py \
    --config residual_18k.yml \
    --weight_pth ../output/swin-18k-residual-q-dropout \
    --data_pth ../data/DNS_CHALLENGE/processed_wav \
    --bit_per_stream 3.0 \
    --device cuda

python evaluate.py \
    --config residual_9k_gan.yml \
    --weight_pth ../output/swin-9k-residual-gan-250k \
    --data_pth ../data/DNS_CHALLENGE/processed_wav \
    --bit_per_stream 1.5 \
    --device cuda 
"""