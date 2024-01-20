import argparse, yaml, os, torch, json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models.codec import SwinAudioCodec
from data import fetch_dataset, make_data_loader
from utils import dict2namespace, manage_checkpoint


def count_codebook_stats(model, dl, max_streams=5, num_vq=6, codebook_size=1024, device="cuda"):

    model = model.to(device)

    vq_counts = {
        f"stream_{S}_group_{G+1}": torch.zeros(codebook_size) for S in range(max_streams) for G in range(num_vq)
    }
    vq_total_counts = {
        f"stream_{S}_group_{G+1}": 0 for S in range(max_streams) for G in range(num_vq)
    }

    for i, input in tqdm(enumerate(dl), total=len(dl), desc="Count Codebook Stats"):

        x = input['audio'].to(device)
        multi_codes, _ = model.encode(x, num_streams=max_streams) 

        for s in range(max_streams):
            stream_s_code = multi_codes[s] # batch_size, group_size, N
            for g in range(num_vq):
                stream_s_group_g_code = stream_s_code[:,g,:]
                vq_counts, vq_total_counts = update_stats(stream_s_group_g_code, s, g, vq_counts, vq_total_counts)

    entropy = {}
    usage = {}
    for key, val in vq_counts.items():
        used_entries = torch.sum(val > 0).item()
        percentage_used = used_entries / codebook_size * 100
        usage[key] = percentage_used
    
        index_probs = val.float() / vq_total_counts[key]
        entropy[key] = -(index_probs * torch.log(index_probs + 1e-10)).sum().item()

    return entropy, usage

def update_stats(code, S, G, vq_counts, vq_total_counts):

    flat_indices = code.flatten()
    num_indices = flat_indices.shape[0]
    index_counts = torch.bincount(flat_indices, minlength=vq_counts["stream_0_group_1"].size(0))

    vq_counts[f"stream_{S}_group_{G+1}"] += index_counts.cpu()
    vq_total_counts[f"stream_{S}_group_{G+1}"] += num_indices
    return vq_counts, vq_total_counts


def init_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default="residual.yml", help='Path to the config file')
    parser.add_argument("--weight_pth", type=str)

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_streams", type=int, default=5)
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    with open(os.path.join('./configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    return args, config


def run(args, config):
    model = SwinAudioCodec(**vars(config.model)).to(args.device)

    ckp = torch.load(f"{args.weight_pth}/best.pt", map_location=args.device)
    new_state_dict = manage_checkpoint(ckp)
    model.load_state_dict(new_state_dict)

    datasets = fetch_dataset("DNS_CHALLENGE", data_dir=config.data.data_dir, in_freq=config.data.in_freq,
                        win_len=config.data.win_len, hop_len=config.data.hop_len, sr=config.data.sr)
    data_loaders = make_data_loader(datasets, 
                                    batch_size={"train": 20, "test": 16}, 
                                    shuffle={"train": False, "test": False}, 
                                    sampler={"train": None, "test": None}, 
                                    num_workers=args.num_worker)

    entropy, usage = count_codebook_stats(
        model, 
        data_loaders[args.split], 
        max_streams=args.num_streams, 
        num_vq=config.model.num_vqs, 
        codebook_size=config.model.codebook_size, 
        device=args.device
    )

    os.makedirs(f"{args.weight_pth}/vq_stats/{args.split}", exist_ok=True)
    print(f"Saving results into {args.weight_pth}/vq_stats/{args.split}")

    print("Entropy: \n", json.dump(entropy, open(f"{args.weight_pth}/vq_stats/{args.split}/entropy.json", "w"), indent=2))
    print("Effective Percentage: \n", json.dump(usage, open(f"{args.weight_pth}/vq_stats/{args.split}/usage.json", "w"), indent=2))




def visualize(entropy_stats, usage_stats, title="VQ Utility Table on Testset"):

    """
        Args:
            entropy_stats: dict of {"stream_S_group_G+1": float, ...}
            usage_stats: dict {"stream_S_group_G+1": float, ...}
    """

    num_stream = int(list(entropy_stats.keys())[-1].split("_")[1])
    num_vq = int(list(entropy_stats.keys())[-1].split("_")[-1])

    entropy_stats_new = {f"stream_{i}":[] for i in range(num_stream+1)}
    usage_stats_new = {f"stream_{i}":[] for i in range(num_stream+1)}
    for i, (key,entropy) in enumerate(entropy_stats.items()):
        s = key.split("_")[1]
        usage = list(usage_stats.values())[i]
        entropy_stats_new[f"stream_{s}"].append(entropy)
        usage_stats_new[f'stream_{s}'].append(usage)

    fig, ax1 = plt.subplots(figsize=(16, 10))
    fig.suptitle(title, fontsize=12)

    ax1.set_xlabel('Percentage Used')
    ax1.set_xlim(0, 120)
    ax2 = ax1.twiny()
    ax2.set_xlim(0, 40)
    ax2.set_xlabel('Entropy')

    bar_width = 0.4
    gap_between_vqs = 0.3
    gap_between_streams = 1

    total_bar_height = num_vq * bar_width + (num_vq - 1) * gap_between_vqs
    total_height = (num_stream+1) * total_bar_height + (num_stream) * gap_between_streams

    y_positions = np.arange(0, total_height, total_bar_height+gap_between_streams)
    for i, (key) in enumerate(entropy_stats_new.keys()):

        entropies, percentages = entropy_stats_new[key], usage_stats_new[key]

        y = y_positions[i] + np.arange(num_vq) * (bar_width + gap_between_vqs)
        ax1.barh(y, percentages, height=bar_width, color='green', edgecolor='white')

        percentage_aligned = [p/3 for p in percentages]
        ax2.barh(y, entropies, left=percentage_aligned, height=bar_width, color='blue', edgecolor='white')

    ax1.set_xlabel('Percentage Used -/1024 (%)')
    ax1.set_ylabel("VQ Stream")
    ax1.set_yticks(y_positions + (total_bar_height - bar_width) / 2)
    ax1.set_yticklabels([f'Stream {i}' for i in range(num_stream+1)])

    ax1.grid(axis="x")
    ax1.legend(['Percentage Used'], loc='upper right')
    ax2.legend(['Entropy'], loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args, config = init_args_and_config()
    run(args, config)


"""
export CUDA_VISIBLE_DEVICES=1
python run_vq_stats.py \
    --config residual_18k.yml \
    --weight_pth ../output/swin-18k-residual \
    --split test \
    --num_streams 6 \
    --num_worker 4 \
    --device cuda

python run_vq_stats.py \
    --config residual_18k_vq_ema.yml \
    --weight_pth ../output/swin-18k-residual-vq-ema \
    --split test \
    --num_streams 6 \
    --num_worker 4 \
    --device cuda

python run_vq_stats.py \
    --config residual_18k_vq_control.yml \
    --weight_pth ../output/swin-18k-residual-vq-control \
    --split test \
    --num_streams 6 \
    --num_worker 4 \
    --device cuda

"""