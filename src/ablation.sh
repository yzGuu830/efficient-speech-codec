
# # 6k w EMA
# accelerate launch main.py \
#     --config residual_6k_vq_ema.yml \
#     --seed 53 \
#     --wb_exp_name swin-6k-residual-ema \
#     --wb_project_name Neural_Speech_Coding \
#     --num_epochs 50 \
#     --lr 1.0e-4 \
#     --train_bs_per_device 18 \
#     --test_bs_per_device 16 \
#     --num_device 2 \
#     --parallel accel \
#     --num_worker 16

# # 6k w Norm
# accelerate launch main.py \
#     --config residual_6k.yml \
#     --seed 53 \
#     --wb_exp_name swin-6k-residual \
#     --wb_project_name Neural_Speech_Coding \
#     --num_epochs 50 \
#     --lr 1.0e-4 \
#     --train_bs_per_device 18 \
#     --test_bs_per_device 16 \
#     --num_device 2 \
#     --parallel accel \
#     --num_worker 16


# # 6k w Norm w dropout
# accelerate launch main.py \
#     --config residual_6k.yml \
#     --seed 53 \
#     --wb_exp_name swin-6k-residual-dropout \
#     --wb_project_name Neural_Speech_Coding \
#     --num_epochs 50 \
#     --lr 1.0e-4 \
#     --train_bs_per_device 9 \
#     --test_bs_per_device 16 \
#     --num_device 4 \
#     --q_dropout_rate .75 \
#     --augment \
#     --parallel accel \
#     --num_worker 32


# # 6k w Norm w GAN
# accelerate launch main.py \
#     --config residual_6k_gan.yml \
#     --seed 53 \
#     --wb_exp_name swin-6k-residual \
#     --wb_project_name Neural_Speech_Coding \
#     --num_epochs 50 \
#     --lr 1.0e-4 \
#     --train_bs_per_device 18 \
#     --test_bs_per_device 16 \
#     --num_device 2 \
#     --parallel accel \
#     --adv_training \
#     --num_worker 16


# 9k w Norm
accelerate launch main.py \
    --config residual_9k.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 16 \
    --num_device 4 \
    --parallel accel \
    --num_worker 32 \
    --save_steps 27000 30000 35000 40000 50000 100000 200000

# 9k w Norm w dropout [bs=40,different setting]
accelerate launch main.py \
    --config residual_9k.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-dropout \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 18 \
    --test_bs_per_device 16 \
    --num_device 2 \
    --q_dropout_rate .5 \
    --parallel accel \
    --num_worker 16

# 9k w Norm w dropout w rvq
accelerate launch main.py \
    --config residual_9k_rvq.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-dropout-rvq \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 18 \
    --test_bs_per_device 16 \
    --num_device 2 \
    --q_dropout_rate 1. \
    --parallel accel \
    --num_worker 16

# 9k w Norm w GAN w dropout [250k]
accelerate launch main.py \
    --config residual_9k_gan.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-gan \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --num_device 4 \
    --q_dropout_rate .5 \
    --parallel accel \
    --adv_training \
    --num_worker 32

# run a final one?
accelerate launch main.py \
    --config residual_9k_gan.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-gan \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --num_device 4 \
    --q_dropout_rate .5 \
    --parallel accel \
    --adv_training \
    --num_worker 32
    --config residual_9k_gan.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-gan \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --num_device 4 \
    --q_dropout_rate .5 \
    --parallel accel \
    --adv_training \
    --num_worker 16