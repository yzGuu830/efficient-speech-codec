from .metrics import EntropyCounter, PESQ, MelSpectrogramDistance, SISDR
from .utils import read_yaml, EvalSet
from esc.models import make_model

from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
import numpy as np

import argparse, torch, json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_folder_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--model_path", type=str, required=True, help="folder contains model configuration and checkpoint")
    parser.add_argument("--save_path", type=str, default=None, help="folder to save test statistics")

    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()

@torch.no_grad()
def eval_epoch(model, eval_loader:DataLoader, 
               metric_funcs:dict, e_counter:EntropyCounter, device: str, bps_per_stream: float,
               num_streams=None, verbose: bool=True):
    model.eval()

    all_perf = {k:[] for k in metric_funcs.keys()}
    all_perf["utilization"] = []
    eval_range = range(num_streams,num_streams+1) if num_streams is not None \
        else range(1, model.max_streams+1) # 1.5kbps -> 9kbps
    for s in eval_range: 
        perf = {k:[] for k in metric_funcs.keys()}
        e_counter.reset_stats(num_streams=s)
        for _, x in tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f"Evaluating Codec at {s*bps_per_stream:.2f}kbps"):
            x = x.to(device)
            outputs = model(**dict(x=x, x_feat=None, num_streams=s))
            recon_x, codes = outputs["recon_audio"], outputs["codes"]

            for k, func in metric_funcs.items():    
                perf[k].extend(func(x, recon_x).tolist())
            e_counter.update(codes)

        for k, v in perf.items():
            all_perf[k].append(round(np.mean(v),4))
        rate, _ = e_counter.compute_utilization()
        perf["utilization"] = [rate]
        all_perf["utilization"].append(rate)

        if verbose:
            print(f"Test Metrics at {s*1.5:.2f}kbps: ", end="")
            print(" | ".join(f"{k}: {np.mean(v):.4f}" for k, v in perf.items()))

    return all_perf

def run(args):
    # Data
    eval_set = EvalSet(args.eval_folder_path)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, collate_fn=default_collate)

    # Metrics
    metric_funcs = {"PESQ": PESQ(), "MelDistance": MelSpectrogramDistance().to(args.device), "SISDR": SISDR().to(args.device)}

    # Model
    cfg = read_yaml(f"{args.model_path}/config.yaml")
    model = make_model(cfg['model'], cfg['model_name'])
    model.load_state_dict(
        torch.load(f"{args.model_path}/model.pth", map_location="cpu")["model_state_dict"],
    )
    model = model.to(args.device)
    e_counter = EntropyCounter(cfg['model']['codebook_size'], num_streams=cfg['model']['max_streams'], 
                               num_groups=cfg['model']['group_size'], device=args.device)

    performances = eval_epoch(
            model, eval_loader, metric_funcs, e_counter, args.device,
            num_streams=None, verbose=True, bps_per_stream=1.5, # evaluate across all bitrates
        )
    
    save_path = args.model_path if args.save_path is None else args.save_path
    json.dump(performances, open(f"{save_path}/perf_stats.json", "w"), indent=2)
    print(f"Test statistics saved into {save_path}/perf_stats.json")


if __name__ == "__main__":
    args = parse_args()
    run(args)


"""
python -m scripts.test \
    --eval_folder_path ../evaluation_set/test \
    --batch_size 12 \
    --model_path ./esc9kbps \
    --device cuda


python -m scripts.test \
    --eval_folder_path ../data/ESC_evaluation/test \
    --batch_size 6 \
    --model_path ../output/csvq_conv_9kbps \
    --device cuda

export CUDA_VISIBLE_DEVICES=1
python -m scripts.test \
    --eval_folder_path ../data/ESC_evaluation/test \
    --batch_size 6 \
    --model_path ../output/rvq_conv_9kbps \
    --device cuda

export CUDA_VISIBLE_DEVICES=2
python -m scripts.test \
    --eval_folder_path ../data/ESC_evaluation/test \
    --batch_size 6 \
    --model_path ../output/rvq_swinT_9kbps \
    --device cuda

"""