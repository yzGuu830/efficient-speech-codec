# adap ablations [250k steps]

# Train from scratch
accelerate launch main.py \
    --config residual_9k.yml \
    --seed 53 \
    --wb_exp_name ours-250k-baseline \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --num_device 4 \
    --q_dropout_rate 1.0 \
    --parallel accel \
    --training_fractions 0.0 0.0 1.0 \
    --num_worker 32

# with Warmup 
accelerate launch --config_file ../custom_accel_config.yaml main.py \
    --config residual_9k.yml \
    --seed 53 \
    --wb_exp_name ours-9k-from-warmup-refine-200k \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --num_device 4 \
    --q_dropout_rate 1.0 \
    --parallel accel \
    --training_fractions 0.2 0.0 0.8 \
    --init_ckpt /root/autodl-fs/swin-9k-residual-gan-ADAP/warmup/checkpoint.pt \
    --num_worker 32

# with Warmup with Kmeans Initialize
python main.py \
    --config residual_9k_kmeans.yml \
    --seed 53 \
    --wb_exp_name ours-250k-warmup-kmeans \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --pretrain_epochs 10 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --parallel accel_train_fast_vq \
    --q_dropout_rate 1.0 \
    --num_worker 32



# Train from warmup then freeze
accelerate launch main.py \
    --config residual_9k.yml \
    --seed 53 \
    --wb_exp_name ours-9k-from-warmup-freeze-50k-refine-200k \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --num_device 4 \
    --q_dropout_rate 1.0 \
    --parallel accel \
    --training_fractions 0.2 0.2 0.6 \
    --init_ckpt /root/autodl-fs/swin-9k-residual-gan-ADAP/warmup/checkpoint.pt \
    --num_worker 32

accelerate launch main.py \
    --config residual_9k.yml \
    --seed 53 \
    --wb_exp_name ours-9k-from-warmup-freeze-50k-refine-200k \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --num_device 4 \
    --q_dropout_rate 1.0 \
    --parallel accel \
    --training_fractions 0.2 0.2 0.6 \
    --init_ckpt /root/autodl-fs/swin-9k-residual-gan-ADAP/warmup/checkpoint.pt \
    --num_worker 32

