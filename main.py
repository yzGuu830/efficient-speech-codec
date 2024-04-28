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
    parser.add_argument("--scheduler_type", default="constant", type=str)
    parser.add_argument("--dropout_rate", type=float, default=1.0)
    
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
accelerate launch main.py \
    --exp_name esc9kbps \
    --config_path ./configs/9kbps_final.yaml
    --wandb_project efficient-speech-codec \
    --lr 1.0e-4 \
    --num_epochs 80 \
    --num_pretraining_epochs 15 \
    --num_devices 4 \
    --dropout_rate 0.75 \
    --save_path ../output \
    --seed 53

"""