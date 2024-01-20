from models.codec import SwinAudioCodec
from utils import PESQ, manage_checkpoint, dict2namespace
from data import fetch_dataset, make_data_loader

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
    def __init__(self, device, config) -> None:
        super().__init__(**vars(config.model))

        self.to(device)
        self.device = device

    def from_pretrain(self, path):
        
        ckp = torch.load(f"{path}/best.pt",map_location=self.device)
        new_state_dict = manage_checkpoint(ckp)
        self.load_state_dict(new_state_dict)
        
        print(f"Pretrained Model {path.split('/')[-1]} Loaded")

def eval_multi_scale(args, config):

    # Model
    codec = SwinAudioCodecEval(args.device, config)
    codec.from_pretrain(args.weight_pth)

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
        obj_scores = []
        for i, input in tqdm(enumerate(test_dl), total=len(test_dl), desc=f"Eval at {s*args.bit_per_stream:.2f}kbps"):
            input['audio'], input['feat'] = input['audio'].to(args.device), input['feat'].to(args.device)
            outputs = codec(**dict(x=input["audio"], x_feat=input["feat"], streams=s, train=False))
            obj_scores.extend(
                [PESQ(input['audio'][j].cpu().numpy(), outputs['recon_audio'][j].cpu().numpy()) for j in range(input['audio'].size(0))]
            )
        performance = np.mean(obj_scores)
        performance_table[f"{s*args.bit_per_stream:.2f}kbps"] = performance
        print(f"{s*args.bit_per_stream:.2f}kbps PESQ: {performance:.3f}")
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
"""