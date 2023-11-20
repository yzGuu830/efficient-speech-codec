import torch, json, argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models.codec import SwinCrossScaleCodec
from data import make_data_loader, fetch_dataset

def calculate_codebook_utility(code, codebook_size=1024):
    flat_indices = code.flatten()
    total_indices = flat_indices.shape[0]
    index_counts = torch.bincount(flat_indices, minlength=codebook_size)

    index_probabilities = index_counts.float() / total_indices
    entropy = -(index_probabilities * torch.log(index_probabilities + 1e-10)).sum()
    used_entries = torch.sum(index_counts > 0).item()
    percentage_used = used_entries / codebook_size * 100
    print(f'Entropy: {entropy.item():.2f}', end="  ")
    print(f'Percentage of VQ entries used: {used_entries}/{codebook_size} = {percentage_used:.2f}%')
    return entropy.item(), percentage_used

def gather_all_code(model, data_loader, num_streams=6):

    all_codes = {f"stream {i}":[] for i in range(num_streams)}
    for i, input in tqdm(enumerate(data_loader), desc="Gathering codes"):

        x = input['audio'].to(device)
        multi_codes, _ = model.encode(x, num_streams=num_streams)
        # multi_codes: [num_streams*(bs,group_size,points)]
        for j in range(num_streams):
            jth_code = multi_codes[j].permute(1,0,2).flatten(1) # [group_size, bs*points]
            all_codes[f"stream {j}"].append(jth_code)

    for key, val in all_codes.items():
        all_codes[key] = torch.cat(val, dim=-1) # [group_size, points*num_data]
    
    return all_codes

def run(model, data_loader, num_streams):

    all_codes = gather_all_code(model, data_loader, num_streams)
    group_size = all_codes["stream 0"].size(0)

    print("Counting")
    all_stats = []
    for j in range(num_streams):
        stream_i_stats = []
        for g in range(group_size): 
            code = all_codes[f"stream {j}"][g:g+1]
            print(f"Evaluate {j+1}-th/{num_streams} stream and {g+1}-th/{group_size} codebook: ")
            entropy, percentage_used = calculate_codebook_utility(code, codebook_size=1024)
            stream_i_stats.append((entropy, percentage_used))
        all_stats.append(stream_i_stats)

    return all_stats

def visualize(all_stats, title="VQ Utility Table on Testset"):

    fig, ax1 = plt.subplots(figsize=(16, 10))
    fig.suptitle(title, fontsize=12)

    ax1.set_xlabel('Percentage Used')
    ax1.set_xlim(0, 110)
    ax2 = ax1.twiny()
    ax2.set_xlim(0, 40)
    ax2.set_xlabel('Entropy')

    bar_width = 0.4

    num_vq = len(all_stats[0])
    num_stream = len(all_stats)

    gap_between_vqs = 0.3
    gap_between_streams = 1

    total_bar_height = num_vq * bar_width + (num_vq - 1) * gap_between_vqs
    total_height = num_stream * total_bar_height + (num_stream - 1) * gap_between_streams

    y_positions = np.arange(0, total_height, total_bar_height+gap_between_streams)
    for i, stream_stats in enumerate(all_stats):
        entropies, percentages = zip(*stream_stats)

        y = y_positions[i] + np.arange(num_vq) * (bar_width + gap_between_vqs)
        ax1.barh(y, percentages, height=bar_width, color='green', edgecolor='white')

        percentage_aligned = [p/2.75 for p in percentages]
        ax2.barh(y, entropies, left=percentage_aligned, height=bar_width, color='blue', edgecolor='white')

    ax1.set_xlabel('Percentage Used -/1024 (%)')
    ax1.set_ylabel("VQ Stream")
    ax1.set_xlim(0, 110)
    ax1.set_yticks(y_positions + (total_bar_height - bar_width) / 2)
    ax1.set_yticklabels([f'Stream {i+1}' for i in range(len(all_stats))])

    ax1.grid(axis="x")
    ax1.legend(['Percentage Used'], loc='upper right')
    ax2.legend(['Entropy'], loc='upper left')

    plt.tight_layout()
    plt.show()

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/scratch/eys9/output/swin-18k-scale-baseline")
    parser.add_argument("--data_dir", type=str, default="/scratch/eys9/data/DNS_CHALLENGE/processed_yz")

    parser.add_argument("--streams", type=int, default=6)
    parser.add_argument("--in_freq", type=int, default=192)

    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--test_bs", type=int, default=16)

    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()

if __name__ == "__main__":
    args = init_args()
    device = args.device

    config = json.load(open(f"{args.model_path}/config.json", "r"))
    config["fuse_net"] = None
    weight = torch.load(f"{args.model_path}/best.pt", map_location=device)
    
    model = SwinCrossScaleCodec(**config)
    model.load_state_dict(weight["model_state_dict"])
    model.to(device)

    dataset = fetch_dataset("DNS_CHALLENGE", data_dir=args.data_dir, in_freq=args.in_freq)
    data_loaders = make_data_loader(dataset, 
                                    batch_size={"train":args.train_bs, "test":args.test_bs}, 
                                    shuffle={"train":False, "test":False}, sampler=None, num_workers=3, verbose=True)

    results = {"train":None, "test":None}
    for split in results.keys():
        dl = data_loaders[split]
        print(f"Evaluate on {split} set")
        all_stats = run(model, dl, num_streams=args.streams)

        results[split] = all_stats

    if args.save_path is None:
        args.save_path = args.model_path

    print(f"Saving evaluated utility stats into {args.save_path}")
    json.dump(results, open(f"{args.save_path}/vq_utility_stats.json", "w"), indent=4)


"""

python vq_utility.py \
    --model_path ../output/swin-18k-scale-baseline \
    --streams 6 \
    --train_bs 16 \
    --test_bs 16 \
    --device cuda


"""