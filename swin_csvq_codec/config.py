import torch.nn as nn
import argparse

def init_args():
    parser = argparse.ArgumentParser()
    ### Save Config
    parser.add_argument("--scalable", action="store_true")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--data_name", type=str, default="DNS_CHALLENGE")
    parser.add_argument("--data_path", type=str, default="/scratch/eys9/data")
    parser.add_argument("--save_path", type=str, default="/scratch/eys9/output")
    parser.add_argument("--num_streams", type=int, default=6)

    ### Swin Config
    parser.add_argument("--init_H", type=int, default=192)
    parser.add_argument("--patch_size", type=tuple, default=(3,2))
    parser.add_argument("--model_depth", type=int, default=5)
    parser.add_argument("--layer_depth", type=int, default=2)
    parser.add_argument("--d_model", type=tuple, default=(18,24,36,48,72))
    parser.add_argument("--num_heads", type=tuple, default=(3,3,6,12,24))
    parser.add_argument("--mlp_ratio", type=float, default=4.)

    ### CSVQ Config
    parser.add_argument("--vq_down_ratio", type=float, default=.75)
    parser.add_argument("--num_overlaps", type=tuple, default=(2,2,2,2,2,2))
    parser.add_argument("--num_groups", type=int, default=6)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--vq_commit", type=float, default=1.)

    ### Training Config
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--train_bs", type=int, default=72)
    parser.add_argument("--test_bs", type=int, default=36)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--plot_interval", type=float, default=.66)
    parser.add_argument("--cosine_warmup", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--clip_max_norm", type=float, default=1.)
    
    return parser.parse_args()

args = init_args()

data_name=args.data_name
data_path=args.data_path
save_path=args.save_path
scalable=args.scalable
pretrain=args.pretrain
num_streams=args.num_streams

lr=args.lr
train_bs=args.train_bs
test_bs=args.test_bs
num_workers=args.num_workers
epochs=args.epochs
cosine_warmup=args.cosine_warmup
plot_interval=args.plot_interval
warmup_epochs=args.warmup_epochs
clip_max_norm=args.clip_max_norm

init_H=args.init_H
patch_size=args.patch_size
model_depth=args.model_depth
layer_depth=args.layer_depth
d_model=args.d_model
num_heads=args.num_heads
window_size=args.window_size
mlp_ratio=args.mlp_ratio
in_channels=2
qkv_bias=True
qk_scale=None
proj_drop=0.
attn_drop=0.
norm_layer=nn.LayerNorm

vq_down_ratio=args.vq_down_ratio
num_overlaps=args.num_overlaps
num_groups=args.num_groups
codebook_size=args.codebook_size
vq_commit=args.vq_commit

win_length=20
hop_length=5
sr=16e3

seed=53
device='cuda'