import yaml, os, argparse
from utils import dict2namespace
from scripts.ddp_trainer import main as ddp_main
from scripts.dp_trainer import main as dp_main
import torch.multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # Default
    parser.add_argument('--config', type=str, required=True, default="residual.yml", help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument("--parallel", default="ddp", type=str)
    parser.add_argument('--wb_exp_name', type=str, default='swin-18k', help='WandB exp name')
    parser.add_argument('--wb_project_name', type=str, default=None, help='WandB project name')
    parser.add_argument("--save_dir", type=str, default="/scratch/eys9/output")

    # Train & Test
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--train_bs_per_device", type=int, default=15)
    parser.add_argument("--test_bs_per_device", type=int, default=4)
    parser.add_argument("--num_device", type=int, default=4)
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--scheduler_type", type=str, default="constant")
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--plot_interval", type=float, default=.66)
    parser.add_argument("--info_steps", type=int, default=5)

    args = parser.parse_args()

    with open(os.path.join('./configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    return args, config



if __name__ == "__main__":
    args, config = parse_args_and_config()

    if args.parallel == "ddp":
        world_size = args.num_device
        mp.spawn(ddp_main, args=(world_size, args, config), nprocs=world_size)

    elif args.parallel == "dp":
        dp_main(args, config)
    

"""
python main.py \
    --config residual_18k.yml \
    --seed 53 \
    --wb_exp_name swin-18k-residual \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 80 \
    --lr 1.0e-4 \
    --train_bs_per_device 60 \
    --test_bs_per_device 16 \
    --num_device 4 \
    --parallel dp \
    --num_worker 4

python main.py \
    --config residual_18k_vq_control.yml \
    --seed 53 \
    --wb_exp_name swin-18k-residual-vq-control \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 80 \
    --lr 1.0e-4 \
    --train_bs_per_device 60 \
    --test_bs_per_device 16 \
    --num_device 4 \
    --parallel dp \
    --num_worker 4

python main.py \
    --config residual_15k.yml \
    --seed 53 \
    --wb_exp_name swin-15k-residual-dac-loss \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 80 \
    --lr 1.0e-4 \
    --train_bs_per_device 60 \
    --test_bs_per_device 40 \
    --num_device 4 \
    --parallel dp \
    --num_worker 4
    
python main.py \
    --config cross_merge_15k.yml \
    --seed 53 \
    --wb_exp_name swin-15k-cross-merge \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 80 \
    --lr 1.0e-4 \
    --train_bs_per_device 60 \
    --test_bs_per_device 40 \
    --num_device 4 \
    --parallel dp \
    --num_worker 4

"""