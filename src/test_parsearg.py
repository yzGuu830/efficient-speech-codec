from models.codec import SwinCrossScaleCodec

from config import args


configs = {
        "patch_size":[args.freq_patch,args.time_patch], "swin_depth": args.swin_depth,
        "swin_heads": args.swin_heads, "window_size": args.window_size, "mlp_ratio": args.mlp_ratio,
        "in_freq": args.in_freq, "h_dims": args.swin_h_dims, "max_streams":args.max_streams, 
        "proj": args.proj, "overlap": args.overlap, "num_vqs": args.num_vqs, "codebook_size": args.codebook_size, 
        "cosine_similarity": args.cosine_sim,
        "mel_nfft": args.mel_nfft, "mel_bins": args.mel_bins, 
        "fuse_net": args.fuse_net, "scalable": args.scalable, 
        "spec_augment": args.spec_augment, "win_len": args.win_len, "hop_len": args.hop_len, "sr": args.sr,
        "vis": True
        }

model = SwinCrossScaleCodec(**configs)


"""

python test_parsearg.py \
    --max_streams 6 \
    --swin_h_dims 45 45 72 96 192 384 \
    --swin_heads 3 3 6 12 24 \
    --proj 8 8 8 8 8 8 \
    --overlap 4 \
    --use_wb \
    --wb_exp_name swin-9k-scale-baseline \
    --scalable \
    --epochs 60 \
    --lr 1.0e-4 \
    --train_bs_per_device 15 \
    --test_bs_per_device 4 \
    --num_device 4 \
    --seed 830

"""