from models.codec import SwinAudioCodec
from utils import manage_checkpoint, dict2namespace
from torch.utils.data import Dataset, DataLoader
from scripts.utils import make_metrics

import argparse
import torch, json, os, yaml, torchaudio
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
        
class EvalSet(Dataset):
    def __init__(self, eval_folder) -> None:
        super().__init__()

        self.source_audios = torch.cat([torchaudio.load(f"{eval_folder}/test/1-1158/clip_{i+1}.wav")[0][:, :-80] for i in range(1158)])

        # Tensors of size [bs=1158, C=1, T=159920]
        assert len(self.source_audios)==1158

    def __len__(self):
        return len(self.source_audios)

    def __getitem__(self, i):
        return {
            "audio": self.source_audios[i:i+1], # [1, T]
        }
    
def collate_fn(batch):
    out = {"audio":[]}
    for b in batch:
        for k, v in b.items():
            out[k].append(v)
    out["audio"] = torch.cat(out["audio"], dim=0)
    return out

def make_dataloader(data_pth, bs, num_worker):
    ds = EvalSet(data_pth)
    dl = DataLoader(ds, batch_size=bs, collate_fn=collate_fn, shuffle=False, num_workers=num_worker)
    exp = next(iter(dl))
    print("Loaded! Num of batches: ", len(dl))
    return dl

def eval_multi_scale(args, config):

    # Model
    codec = SwinAudioCodecEval(config)
    codec.from_pretrain(args.weight_pth)
    codec = codec.to(args.device)

    # Metrics
    metrics = make_metrics(args.device)

    # Data
    test_dl = make_dataloader(args.data_pth, bs=12, num_worker=args.num_worker)
    
    # Evaluate loop
    performance_table = {}
    for s in range(1, 7):
        test_perf = {"Test_PESQ":[], "Test_MelDist":[], "Test_STFTDist":[], "Test_SNR": []}
        for i, input in tqdm(enumerate(test_dl), total=len(test_dl), desc=f"Eval at {s*args.bit_per_stream:.2f}kbps"):
            input['audio'], input['feat'] = input['audio'].to(args.device), None
            outputs = codec(**dict(x=input["audio"], x_feat=input["feat"], streams=s, train=False))
            for k, m in metrics.items():
                if k in test_perf:
                    test_perf[k].extend(m(input["audio"], outputs['recon_audio']).tolist())

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

python evaluate.py \
    --config residual_9k_gan.yml \
    --weight_pth ../output/swin-9k-residual-gan-ADAP/refine \
    --data_pth ../data/DNS_CHALLENGE/processed_wav \
    --bit_per_stream 1.5 \
    --device cuda
    
python evaluate.py \
    --config residual_9k.yml \
    --weight_pth /root/autodl-fs/output/swin-9k-residual-dropout-plaw-refine \
    --data_pth /root/autodl-fs/data/DNS_CHALLENGE/processed_wav \
    --bit_per_stream 1.5 \
    --device cuda
"""