import argparse

from scripts.trainer_no_adv import main as train_no_adv
from utils import read_yaml, dict2namespace

def parse_args_config():
    parser = argparse.ArgumentParser()
    
    # Experimental Setups
    parser.add_argument("--exp_name", default="esc9kbps", type=str)
    parser.add_argument("--wandb_project", default=None, type=str)
    parser.add_argument("--lr", default=1.e-4, type=float)
    parser.add_argument("--num_epochs", default=80, type=int)
    parser.add_argument("--num_pretraining_epochs", default=10, type=int)
    parser.add_argument("--num_devices", default=4, type=int)
    parser.add_argument("--num_warmup_steps", default=0, type=int)
    parser.add_argument("--val_metric", default="PESQ", type=str)
    parser.add_argument("--scheduler_type", default="constant", type=str)
    parser.add_argument("--dropout_rate", type=float, default=1.0)
    parser.add_argument("--pretrain_ckp", type=str, default=None)
    
    parser.add_argument("--log_steps", default=5, type=int)
    parser.add_argument("--save_path", default="./output", type=str)
    parser.add_argument("--config_path", default="./configs/9kbps_final.yaml")
    parser.add_argument("--seed", default=1234, type=int)

    args = parser.parse_args()    
    config = dict2namespace(read_yaml(args.config_path))
    return args, config


if __name__ == "__main__":
    args, config = parse_args_config()
    train_no_adv(args, config)


"""
# ESC Final
accelerate launch main.py \
    --exp_name esc9kbps \
    --config_path ./configs/9kbps_final.yaml \
    --wandb_project efficient-speech-codec \
    --lr 1.0e-4 \
    --num_epochs 80 \
    --num_pretraining_epochs 15 \
    --num_devices 4 \
    --dropout_rate 0.75 \
    --save_path ../output \
    --seed 53


# CSVQ + SwinT
accelerate launch main.py \
    --exp_name csvq_swinT_9kbps \
    --config_path ./configs/ablations/9kbps_csvq_swinT.yaml \
    --wandb_project ESC-EMNLP-2025 \
    --lr 1.0e-4 \
    --num_epochs 50 \
    --num_pretraining_epochs 5 \
    --num_devices 4 \
    --dropout_rate 0.75 \
    --save_path ../output \
    --seed 53

# CSVQ + CNN 
accelerate launch main.py \
    --exp_name csvq_conv_9kbps \
    --config_path ./configs/ablations/9kbps_csvq_conv.yaml \
    --wandb_project ESC-EMNLP-2025 \
    --lr 1.0e-4 \
    --num_epochs 50 \
    --num_pretraining_epochs 5 \
    --num_devices 4 \
    --dropout_rate 0.75 \
    --save_path ../output \
    --seed 53

accelerate launch main.py \
    --exp_name csvq_conv_9kbps \
    --config_path ./configs/ablations/9kbps_csvq_conv.yaml \
    --wandb_project ESC-EMNLP-2025 \
    --lr 1.0e-4 \
    --num_epochs 50 \
    --num_pretraining_epochs 5 \
    --num_devices 4 \
    --dropout_rate 0.75 \
    --pretrain_ckp ../output/csvq_conv_9kbps/pretrained.pth \
    --save_path ../output \
    --seed 53

# RVQ + CNN 
accelerate launch main.py \
    --exp_name rvq_conv_9kbps \
    --config_path ./configs/ablations/9kbps_rvq_conv.yaml \
    --wandb_project ESC-EMNLP-2025 \
    --lr 1.0e-4 \
    --num_epochs 50 \
    --num_pretraining_epochs 5 \
    --num_devices 2 \
    --dropout_rate 0.75 \
    --save_path ../output \
    --seed 53

# RVQ + SwinT 
accelerate launch main.py \
    --exp_name rvq_swinT_9kbps \
    --config_path ./configs/ablations/9kbps_rvq_swinT.yaml \
    --wandb_project ESC-EMNLP-2025 \
    --lr 1.0e-4 \
    --num_epochs 50 \
    --num_pretraining_epochs 10 \
    --num_devices 2 \
    --dropout_rate 0.75 \
    --save_path ../output \
    --seed 53

# accelerate launch main.py \
#     --exp_name rvq_swinT_9kbps \
#     --config_path ./configs/ablations/9kbps_rvq_swinT.yaml \
#     --wandb_project ESC-EMNLP-2025 \
#     --lr 1.0e-4 \
#     --num_epochs 50 \
#     --num_pretraining_epochs 10 \
#     --num_devices 2 \
#     --dropout_rate 0.75 \
#     --save_path ../output \
#     --pretrain_ckp ../output/rvq_swinT_9kbps/pretrained.pth \
#     --seed 53

"""