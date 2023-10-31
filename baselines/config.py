import argparse

def init_args():
    parser = argparse.ArgumentParser()
    # Model Config
    parser.add_argument("--in_freq", type=int, default=192)
    parser.add_argument("--max_streams", type=int, default=6)
    # parser.add_argument("--h_dims", type=list, default=[36,48])
    parser.add_argument("--h_dims", type=list, default=[16,16,24,24,32,64])
    parser.add_argument("--use_tf", action='store_true')
    parser.add_argument("--fuse_net", action='store_true')
    parser.add_argument("--scalable", action="store_true")
    parser.add_argument("--spec_augment", action="store_true")

    # swin parameter
    parser.add_argument("--freq_patch", type=int, default=3)
    parser.add_argument("--time_patch", type=int, default=2)
    # parser.add_argument("--swin_h_dims", type=list, default=[16,16,32,32,64,128])
    parser.add_argument("--swin_h_dims", type=list, default=[45, 45, 72, 96, 192, 384])
    # parser.add_argument("--swin_h_dims", type=list, default=[36,72,144,144,192,384])
    parser.add_argument("--swin_depth", type=int, default=2)
    parser.add_argument("--swin_heads", type=int, default=[3, 3, 6, 12, 24])
    # parser.add_argument("--swin_heads", type=int, default=3)
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=4.)    
    
    parser.add_argument("--proj", type=int, default=[4,4,2,2,2,2])
    # parser.add_argument("--proj", type=int, default=2)
    parser.add_argument("--overlap", type=int, default=4)
    parser.add_argument("--num_vqs", type=int, default=6)
    parser.add_argument("--codebook_size", type=int, default=1024)

    parser.add_argument("--mel_nfft", type=int, default=2048)
    parser.add_argument("--mel_bins", type=int, default=64)

    parser.add_argument("--win_len", type=int, default=20)
    parser.add_argument("--hop_len", type=int, default=5)
    parser.add_argument("--sr", type=int, default=16000)
    
    # Train Config
    parser.add_argument("--data_dir", type=str, default="/scratch/eys9/data/DNS_CHALLENGE/processed_yz")
    parser.add_argument("--save_dir", type=str, default="/scratch/eys9/output")
    parser.add_argument("--use_wb", action="store_true")
    parser.add_argument("--wb_exp_name", type=str, default="conv-18k")
    
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--train_bs_per_device", type=int, default=64)
    parser.add_argument("--test_bs_per_device", type=int, default=24)
    parser.add_argument("--num_device", type=int, default=4)

    parser.add_argument("--plot_interval", type=float, default=.66)
    parser.add_argument("--info_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=53)
    
    return parser.parse_args()

args = init_args()