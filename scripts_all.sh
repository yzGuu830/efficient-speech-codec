## Training Final Models
accelerate launch main.py \
    --exp_name esc-base-non-adv \
    --config_path ./configs/9kbps_esc_base.yaml \
    --wandb_project efficient-speech-codec \
    --lr 1.0e-4 \
    --num_epochs 80 \
    --num_pretraining_epochs 15 \
    --num_devices 4 \
    --dropout_rate 0.75 \
    --save_path ../output \
    --seed 53

accelerate launch main.py \
    --exp_name esc-base-adv \
    --adv_training \
    --config_path ./configs/9kbps_esc_base_adv.yaml \
    --wandb_project efficient-speech-codec \
    --lr 1.0e-4 \
    --num_epochs 80 \
    --num_pretraining_epochs 15 \
    --num_devices 4 \
    --dropout_rate 0.75 \
    --save_path ../output \
    --seed 53

# accelerate launch main.py \
#     --exp_name esc-base-post-adv \
#     --adv_training \
#     --pretrain_ckp ../esc9kbps_base_non_adversarial/model.pth \
#     --config_path ./configs/9kbps_esc_base_adv.yaml \
#     --wandb_project efficient-speech-codec \
#     --lr 1.0e-4 \
#     --num_epochs 20 \
#     --num_pretraining_epochs 0 \
#     --num_devices 4 \
#     --dropout_rate 0.75 \
#     --save_path ../output \
#     --seed 53

accelerate launch main.py \
    --exp_name esc-large-non-adv \
    --config_path ./configs/9kbps_esc_large.yaml \
    --wandb_project efficient-speech-codec \
    --lr 1.0e-4 \
    --num_epochs 80 \
    --num_pretraining_epochs 15 \
    --num_devices 4 \
    --dropout_rate 0.75 \
    --save_path ../output \
    --seed 53


## Method Ablations
accelerate launch main.py \
    --exp_name csvq+swinT \
    --config_path ./configs/ablations/9kbps_csvq_swinT.yaml \
    --wandb_project efficient-speech-codec \
    --lr 1.0e-4 \
    --num_epochs 50 \
    --num_pretraining_epochs 5 \
    --num_devices 4 \
    --dropout_rate 0.75 \
    --save_path ../output \
    --seed 53

accelerate launch main.py \
    --exp_name csvq+conv_9kbps \
    --config_path ./configs/ablations/9kbps_csvq_conv.yaml \
    --wandb_project efficient-speech-codec \
    --lr 1.0e-4 \
    --num_epochs 50 \
    --num_pretraining_epochs 5 \
    --num_devices 4 \
    --dropout_rate 0.75 \
    --save_path ../output \
    --seed 53

accelerate launch main.py \
    --exp_name rvq+swinT \
    --config_path ./configs/ablations/9kbps_rvq_swinT.yaml \
    --wandb_project efficient-speech-codec \
    --lr 1.0e-4 \
    --num_epochs 50 \
    --num_pretraining_epochs 5 \
    --num_devices 2 \
    --dropout_rate 0.75 \
    --save_path ../output \
    --seed 53

accelerate launch main.py \
    --exp_name rvq+conv \
    --config_path ./configs/ablations/9kbps_rvq_conv.yaml \
    --wandb_project efficient-speech-codec \
    --lr 1.0e-4 \
    --num_epochs 50 \
    --num_pretraining_epochs 5 \
    --num_devices 4 \
    --dropout_rate 0.75 \
    --save_path ../output \
    --seed 53

accelerate launch main.py \
    --exp_name csvq+swinT_w/o_pretraining \
    --config_path ./configs/ablations/9kbps_csvq_swinT.yaml \
    --wandb_project efficient-speech-codec \
    --lr 1.0e-4 \
    --num_epochs 50 \
    --num_pretraining_epochs 0 \
    --num_devices 2 \
    --dropout_rate 0.75 \
    --save_path ../output \
    --seed 53