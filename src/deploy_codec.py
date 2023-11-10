import torch.nn as nn
from models.codec import *
from utils import show_and_save

import argparse
import torch, torchaudio
import os, json

def init_args():
    parser = argparse.ArgumentParser()
    ### Save Config
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--audio_path", type=str)
    parser.add_argument("--streams", type=int, default=6)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()

class Audio_Codec(ConvCrossScaleCodec):
    def __init__(self, device, config) -> None:
        super().__init__(**config)

        self.to(device)
        self.device = device

    def from_pretrain(self, path):
        
        weights = torch.load(f"{path}/best.pt",map_location=self.device)['model_state_dict']
        
        self.load_state_dict(weights)
        print(f"Pretrained Model {path.split('/')[-1]} Loaded")

    def reconstruct(self, audio_path, streams, save_path):
        self.eval()

        x, sr = torchaudio.load(audio_path)
        x = x[:, :-80].to(self.device)

        outputs = self.test_one_step(x, x_feat=None, streams=streams)
        x_ = outputs["recon_audio"]

        if not os.path.exists(save_path): os.makedirs(save_path)
        show_and_save(x, x_, save_path, use_wb=False)
        print(f"Saving into {save_path}")

def main():
    args = init_args()
    config = json.load(open(f"{args.model_path}/config.json", 'r'))

    codec = Audio_Codec(args.device, config)
        
    codec.from_pretrain(args.model_path)
    codec.reconstruct(args.audio_path, 
                      args.streams, 
                      save_path=f"{args.save_path}/{args.streams*3}kbps")
    
    return


if __name__ == "__main__":
    main()

"""
python deploy_codec.py \
    --device cuda \
    --audio_path /hpc/home/eys9/dns/test_audios/english_instance503.wav \
    --save_path /scratch/eys9/exps/causal-conv-18k \
    --model_path /scratch/eys9/output/causal-conv-18k \
    --streams 6

python deploy_codec.py \
    --device cuda \
    --audio_path /hpc/home/eys9/dns/test_audios/english_instance503.wav \
    --save_path /scratch/eys9/exps/causal-conv-18k-merge \
    --model_path /scratch/eys9/output/causal-conv-18k-merge \
    --streams 6

    
python deploy_codec.py \
    --device cuda \
    --audio_path /hpc/home/eys9/dns/test_audios/english_instance503.wav \
    --save_path /scratch/eys9/exps/causal-conv-6k-merge \
    --model_path /scratch/eys9/output/causal-conv-6k-merge \
    --streams 2

python deploy_codec.py \
    --device cuda \
    --audio_path /hpc/home/eys9/dns/test_audios/english_instance503.wav \
    --save_path /scratch/eys9/exps/causal-conv-3k-merge \
    --model_path /scratch/eys9/output/causal-conv-3k-merge \
    --streams 1
"""