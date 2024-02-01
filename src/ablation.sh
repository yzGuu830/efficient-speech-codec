# warmup 9k
accelerate launch main.py \
    --config residual_9k.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-warmup \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 10 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 30 \
    --num_device 4 \
    --warmup_training \ 
    --parallel accel \
    --num_worker 36

# 9k w Norm
accelerate launch main.py \
    --config residual_9k.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 30 \
    --num_device 4 \
    --parallel accel \
    --num_worker 32

# 9k w Norm w Dropout [done]
accelerate launch main.py \
    --config residual_9k.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-dropout-plaw \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 30 \
    --num_device 4 \
    --q_dropout_rate .5 \
    --parallel accel \
    --num_worker 36

# 9k w Norm w Dropout from warmup [done]
accelerate launch main.py \
    --config residual_9k.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-dropout-plaw \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 30 \
    --num_device 4 \
    --from_warmup ../output/swin-9k-residual-warmup \
    --q_dropout_rate .5 \
    --parallel accel \
    --num_worker 36

# 9k w EMA w Dropout
accelerate launch main.py \
    --config residual_9k_vq_ema.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-dropout-plaw-ema \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 30 \
    --num_device 4 \
    --q_dropout_rate .5 \
    --parallel accel \
    --num_worker 36

# 9k w Norm w Dropout w rvq
accelerate launch main.py \
    --config residual_9k_rvq.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-dropout-plaw-rvq \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 30 \
    --num_device 4 \
    --q_dropout_rate .5 \
    --parallel accel \
    --num_worker 36

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
    --num_worker 16