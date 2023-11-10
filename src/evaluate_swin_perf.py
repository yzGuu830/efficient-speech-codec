import torch.nn as nn
from models.codec import *
from utils import show_and_save, PESQ
from data import fetch_dataset, make_data_loader

import argparse
import torch, torchaudio, json
from tqdm import tqdm

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)

    # parser.add_argument("--scalable", action="store_true")
    # parser.add_argument("--streams", type=int, default=6)

    parser.add_argument("--save_path", type=str)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()

class Swin_Codec(SwinCrossScaleCodec):
    def __init__(self, device, config) -> None:
        super().__init__(**config)

        self.to(device)
        self.device = device

    def from_pretrain(self, path):
        
        weights = torch.load(f"{path}/best.pt",map_location=self.device)['model_state_dict']
        
        self.load_state_dict(weights)
        print(f"Pretrained Model {path.split('/')[-1]} Loaded")

if __name__ == "__main__":
    args = init_args()
    config = json.load(open(f"{args.model_path}/config.json", 'r'))
    codec = Swin_Codec(args.device, config)   
    codec.from_pretrain(args.model_path)

    dataset = fetch_dataset("DNS_CHALLENGE", data_dir=args.data_path, in_freq=config["in_freq"])
    data_loaders = make_data_loader(dataset, 
                                   batch_size={"train":40, "test":16}, 
                                   shuffle={"train":True, "test":False})
    test_dataloader = data_loaders["test"]

    performance_table = {}
    for s in range(1, 7):
        obj_scores = []
        for i, input in tqdm(enumerate(test_dataloader)):
            input['audio'], input['feat'] = input['audio'].cuda(), input['feat'].cuda()
            outputs = codec(**dict(x=input["audio"], x_feat=input["feat"], streams=s, train=False))
            obj_scores.extend(
                [PESQ(input['audio'][j].cpu().numpy(), outputs['recon_audio'][j].cpu().numpy()) for j in range(input['audio'].size(0))]
            )
        performance = np.mean(obj_scores)
        performance_table[f"{s*3}kbps"] = performance
        print(f"{s*3}kbps PESQ: {performance}")
    print("Saving Full Performance Table into ", f"{args.model_path}/performance.json")
    json.dump(performance, open(f"{args.model_path}/performance.json", 'w'), indent=4)
    
"""
python evaluate_swin_perf.py \
    --model_path /scratch/eys9/output/swin-18k-scale-baseline \
    --data_path /scratch/eys9/data/DNS_CHALLENGE/processed_yz
"""