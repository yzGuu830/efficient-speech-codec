
# 6k w Norm w dropout
accelerate launch main.py \
    --config residual_6k.yml \
    --seed 53 \
    --wb_exp_name swin-6k-residual-dropout \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 80 \
    --lr 1.0e-4 \
    --train_bs_per_device 18 \
    --test_bs_per_device 16 \
    --scheduler_type cosine_warmup \
    --warmup_steps 15000 \
    --num_device 2 \
    --parallel accel \
    --q_dropout_rate .5 \
    --num_worker 16

# 6k w Norm w dropout w GAN
accelerate launch main.py \
    --config residual_6k_gan.yml \
    --seed 53 \
    --wb_exp_name swin-6k-residual \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 18 \
    --test_bs_per_device 16 \
    --scheduler_type cosine_warmup \
    --warmup_steps 15000 \
    --num_device 2 \
    --parallel accel \
    --q_dropout_rate .5 \
    --adv_training \
    --num_worker 16