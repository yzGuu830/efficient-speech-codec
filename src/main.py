import yaml, os, argparse
from utils import dict2namespace
from scripts.ddp_trainer import main as ddp_main
from scripts.dp_trainer import main as dp_main
from scripts.trainer_with_adv import main as dp_main_adv
from scripts.trainer_no_adv import main as dp_main_no_adv
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
    parser.add_argument("--save_dir", type=str, default="/root/autodl-fs/output")
    parser.add_argument("--adv_training", action="store_true")
    parser.add_argument("--q_dropout_rate", type=float, default=1.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--trans_on_cpu", action="store_true")
    parser.add_argument("--training_fractions", nargs="+", default=[0.125,0.625,0.25], type=float)
    parser.add_argument("--save_steps", nargs="+", type=int, default=[50000, 100000, 200000, 250000, 350000, 400000])

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
    parser.add_argument("--info_steps", type=int, default=20)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--init_ckpt", default=None, type=str)

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

    elif args.parallel == 'accel': # Use accelerate library
        if args.adv_training:
            dp_main_adv(args, config)
        else:
            dp_main_no_adv(args, config)

"""
## Baseline ## [done]
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

## Re-span to compare with DAC ## [to do last]
# accelerate launch main.py \
#     --config residual_9k.yml \
#     --seed 53 \
#     --wb_exp_name swin-9k-residual \
#     --wb_project_name Neural_Speech_Coding \
#     --num_epochs 55 \
#     --lr 1.0e-4 \
#     --train_bs_per_device 20 \
#     --test_bs_per_device 16 \
#     --num_device 2 \
#     --parallel accel \
#     --q_dropout_rate .5 \
#     --num_worker 16
accelerate launch main.py \
    --config residual_6k.yml \
    --seed 53 \
    --wb_exp_name swin-6k-residual \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 60 \
    --lr 1.0e-4 \
    --train_bs_per_device 20 \
    --test_bs_per_device 16 \
    --scheduler_type cosine_warmup \
    --warmup_steps 15000 \
    --num_device 2 \
    --parallel accel \
    --q_dropout_rate .5 \
    --num_worker 16

accelerate launch main.py \
    --config residual_6k_gan.yml \
    --seed 53 \
    --wb_exp_name swin-6k-residual-gan \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 60 \
    --lr 1.0e-4 \
    --train_bs_per_device 20 \
    --test_bs_per_device 16 \
    --scheduler_type cosine_warmup \
    --warmup_steps 15000 \
    --num_device 2 \
    --parallel accel \
    --q_dropout_rate .5 \
    --num_worker 16

## Ablation on VQ training approach ## [done]
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
    --config residual_18k_vq_ema.yml \
    --seed 53 \
    --wb_exp_name swin-18k-residual-vq-ema \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 80 \
    --lr 1.0e-4 \
    --train_bs_per_device 60 \
    --test_bs_per_device 16 \
    --num_device 4 \
    --parallel dp \
    --num_worker 16


## Ablation on GAN effects on ours ## [in progress]
accelerate launch main.py \
    --config residual_18k_gan.yml \
    --seed 53 \
    --wb_exp_name swin-18k-residual-gan \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 30 \
    --lr 1.0e-4 \
    --train_bs_per_device 10 \
    --test_bs_per_device 8 \
    --num_device 2 \
    --parallel accel \
    --adv_training \
    --num_worker 16


## Ablation on Qunatization Dropout ##
accelerate launch main.py \
    --config residual_18k.yml \
    --seed 53 \
    --wb_exp_name swin-18k-residual-q-dropout \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 55 \
    --lr 1.0e-4 \
    --train_bs_per_device 20 \
    --test_bs_per_device 8 \
    --num_device 2 \
    --parallel accel \
    --q_dropout_rate 0.5 \
    --num_worker 16

"""