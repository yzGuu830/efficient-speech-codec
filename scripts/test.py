from scripts.metrics import EntropyCounter, PESQ, MelSpectrogramDistance, SISDR
from models import make_model
from utils import read_yaml

from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm
import numpy as np

import torchaudio, glob, argparse, torch, json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_folder_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--model_path", type=str, required=True, help="folder contains model configuration and checkpoint")
    parser.add_argument("--save_path", type=str, default=None, help="folder to save test statistics")

    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()

class EvalSet(Dataset):
    def __init__(self, eval_folder_path) -> None:
        super().__init__()
        self.testset_files = glob.glob(f"{eval_folder_path}/*.wav")
        
    def __len__(self):
        return len(self.testset_files)

    def __getitem__(self, i):
        x, _ = torchaudio.load(self.testset_files[i])
        return x[0, :-80]

@torch.no_grad()
def eval_epoch(model, eval_loader:DataLoader, 
               metric_funcs:dict, e_counter:EntropyCounter, device: str,
               num_streams=None, verbose: bool=True):
    model.eval()

    all_perf = {k:[] for k in metric_funcs.keys()}
    all_perf["utilization"] = []
    eval_range = range(num_streams,num_streams+1) if num_streams is not None \
        else range(1, model.max_streams+1) # 1.5kbps -> 9kbps
    for s in eval_range: 
        perf = {k:[] for k in metric_funcs.keys()}
        e_counter.reset_stats(num_streams=s)
        for _, x in tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f"Evaluating Codec at {s*1.5:.2f}kbps"):
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
    model = make_model(read_yaml(f"{args.model_path}/config.yaml")['model'])
    model.load_state_dict(
        torch.load(f"{args.model_path}/model.pth", map_location="cpu")["model_state_dict"],
    )
    model = model.to(args.device)
    e_counter = EntropyCounter(model.quantizers[0].codebook_size, num_streams=model.max_streams, 
                               num_groups=model.quantizers[0].num_vqs, device=args.device)

    performances = eval_epoch(
            model, eval_loader, metric_funcs, e_counter, args.device,
            num_streams=None, verbose=True # evaluate across all bitrates
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

"""