# Number of steps: 400k steps (80epochs)
# Scheduler during fine-training: 1e-4 Exponential LR decay 
# Warmup: 75k steps; 

# Ours baseline [Train from scratch (TODO LAST)]
accelerate launch main.py \
    --config residual_9k_kmeans.yml \
    --seed 53 \
    --wb_exp_name ours-final-baseline \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 80 \
    --pretrain_epochs 0 \
    --scheduler_type exponential_decay \ # needs to modify [first 15 epochs shall be warmuped lr]
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --parallel accel_train_fast_vq \
    --q_dropout_rate 0.75 \
    --num_worker 36

# Ours baseline + Warmup [75k steps] (TODO LAST)
accelerate launch main.py \
    --config residual_9k_random_select.yml \
    --seed 53 \
    --wb_exp_name ours-final-warmup \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 80 \
    --pretrain_epochs 15 \
    --init_ckpt ../output/ours-final-warmup-kmeans/pretrain/checkpoint.pt \
    --scheduler_type exponential_decay \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --parallel accel_train_fast_vq \
    --q_dropout_rate 0.75 \
    --num_worker 36

# Ours baseline + Warmup + Initialization 
accelerate launch main.py \
    --config residual_9k_kmeans.yml \
    --seed 53 \
    --wb_exp_name ours-final-warmup-kmeans \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 80 \
    --pretrain_epochs 15 \
    --scheduler_type exponential_decay \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --parallel accel_train_fast_vq \
    --q_dropout_rate 0.75 \
    --num_worker 36

# Ours baseline + Warmup + Initialization + GAN
accelerate launch main.py \
    --config residual_9k_kmeans.yml \
    --adv_training \
    --seed 53 \
    --wb_exp_name ours-final-warmup-kmeans-gan \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 80 \
    --pretrain_epochs 15 \
    --init_ckpt ../output/ours-final-warmup-kmeans/pretrain/checkpoint.pt \
    --scheduler_type exponential_decay \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --parallel accel_train_fast_vq \
    --q_dropout_rate 0.75 \
    --num_worker 36