import argparse

from scripts.trainer_no_adv import main as train_no_adv
from scripts.trainer_adv import main as train_adv
from scripts.utils import read_yaml, dict2namespace

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
    parser.add_argument("--adv_training", default=False, action="store_true")
    parser.add_argument("--pretrain_ckp", type=str, default=None)
    
    parser.add_argument("--log_steps", default=5, type=int)
    parser.add_argument("--save_path", default="./output", type=str)
    parser.add_argument("--config_path", default="./configs/9kbps_esc_base.yaml")
    parser.add_argument("--seed", default=1234, type=int)

    args = parser.parse_args()    
    config = dict2namespace(read_yaml(args.config_path))
    return args, config


if __name__ == "__main__":
    args, config = parse_args_config()
    if args.adv_training:
        train_adv(args, config)
    else:
        train_no_adv(args, config)
        