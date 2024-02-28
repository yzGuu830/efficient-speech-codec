# fsq 9k
accelerate launch main.py \
    --config residual_9k_fsq.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-FSQ \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --num_device 4 \
    --q_dropout_rate .5 \
    --parallel accel \
    --training_fractions 0.2 0.0 0.8 \
    --init_ckpt /root/autodl-fs/swin-9k-residual-gan-ADAP/warmup/checkpoint.pt \
    --num_worker 32

# adap ablations [warmup + refine]
accelerate launch main.py \
    --config residual_9k.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-warmup-refine \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --num_device 4 \
    --q_dropout_rate .5 \
    --parallel accel \
    --training_fractions 0.2 0.0 0.8 \
    --init_ckpt /root/autodl-fs/swin-9k-residual-gan-ADAP/warmup/checkpoint.pt \
    --num_worker 32

accelerate launch --config_file ../custom_accel_config.yaml main.py \
    --config residual_9k.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-warmup-refine \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --num_device 4 \
    --q_dropout_rate .5 \
    --parallel accel \
    --training_fractions 0.2 0.0 0.8 \
    --init_ckpt /root/autodl-fs/swin-9k-residual-gan-ADAP/warmup/checkpoint.pt \
    --num_worker 32

# inverse csvq 9k 
accelerate launch main.py \
    --config residual_9k_res_csvq.yml \
    --seed 53 \
    --wb_exp_name swin-9k-res-csvq \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --num_device 4 \
    --q_dropout_rate .5 \
    --warmup_epochs 10 \
    --parallel accel_no_adap \
    --num_worker 32

# inverse csvq 9k w progressive training
accelerate launch main.py \
    --config residual_9k_pro.yml \
    --seed 53 \
    --wb_exp_name swin-9k-res-csvq-progressive \
    --wb_project_name Neural_Speech_Coding \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --num_device 4 \
    --q_dropout_rate .5 \
    --warmup_epochs 0 \
    --parallel accel_pro \
    --num_worker 32

# accelerate launch --config_file ../custom_accel_config.yaml main.py \
#     --config residual_9k_pro.yml \
#     --seed 53 \
#     --wb_exp_name swin-9k-res-csvq-progressive \
#     --lr 1.0e-4 \
#     --train_bs_per_device 9 \
#     --test_bs_per_device 12 \
#     --num_device 4 \
#     --q_dropout_rate .5 \
#     --warmup_epochs 0 \
#     --parallel accel_pro \
#     --num_worker 32

# progressive 9k 400k
accelerate launch main.py \
    --config residual_9k_pro.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-progressive \
    --wb_project_name Neural_Speech_Coding \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 12 \
    --num_device 4 \
    --q_dropout_rate .5 \
    --parallel accel_pro \
    --eval_every epoch \
    --num_worker 32

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

# 9k w Norm w Dropout from warmup [freeze encoder] [done]
accelerate launch main.py \
    --config residual_9k.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-dropout-plaw-freeze \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 50 \
    --lr 1.0e-4 \
    --train_bs_per_device 9 \
    --test_bs_per_device 30 \
    --num_device 4 \
    --from_warmup ../output/swin-9k-residual-warmup \
    --q_dropout_rate .5 \
    --parallel accel \
    --freeze_encoder_layers \
    --num_worker 36
# refine stage
accelerate launch main.py \
    --config residual_9k.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-dropout-plaw-refine \
    --wb_project_name Neural_Speech_Coding \
    --num_epochs 20 \
    --lr 3.0e-5 \
    --train_bs_per_device 9 \
    --test_bs_per_device 30 \
    --num_device 4 \
    --from_warmup ../output/swin-9k-residual-dropout-plaw-freeze \
    --q_dropout_rate .5 \
    --parallel accel \
    --num_worker 36

# 9k w Norm w Dropout w Regularization from warmup [done]
accelerate launch main.py \
    --config residual_9k_reg.yml \
    --seed 53 \
    --wb_exp_name swin-9k-residual-dropout-plaw-freeze-reg \
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


