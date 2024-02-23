import numpy as np
import torch, transformers
import sys
sys.path.append("../")

from models.codec import SwinAudioCodec
from models.losses.metrics import PESQ, MelDistance, SISDRLoss, STFTDistance, PSNR, SNR

def quantization_dropout(dropout_rate: float, max_streams: int):
    """
    Args:
        dropout_rate: probability that applies quantization dropout 
        max_streams: maximum number of streams codec can take
        returns: sampled number of streams for current batch
    """
    if dropout_rate != 1.0:
        # Do Random Sample N w prob dropout_rate
        do_sample = np.random.choice([0, 1], p=[1-dropout_rate, dropout_rate])
        if do_sample: 
            streams = np.random.randint(1, max_streams+1)
        else:
            streams = max_streams
    else:
        # Do not apply quantization dropout
        streams = np.random.randint(1, max_streams+1)

    return streams


def freeze_layers(generator, part="encoder"):
    assert part in ["encoder", "decoder", "quantizer"]
    if part == "encoder":
        for param in generator.encoder.parameters():
            param.requires_grad = False
    elif part == "decoder":
        for param in generator.decoder.parameters():
            param.requires_grad = False
    elif part == "quantizer":
        for param in generator.quantizer.parameters():
            param.requires_grad = False

    trainable_params = [param for name, param in generator.named_parameters() if part not in name]
    return generator, trainable_params

def unfreeze_layers(generator, part="encoder"):
    assert part in ["encoder", "decoder", "quantizer"]
    if part == "encoder":
        for param in generator.encoder.parameters():
            param.requires_grad = True
    elif part == "decoder":
        for param in generator.decoder.parameters():
            param.requires_grad = True
    elif part == "quantizer":
        for param in generator.quantizer.parameters():
            param.requires_grad = True
    return generator

def maintain_stage(batch_idx, adap_training_steps):
    """
    Args:
        batch_idx: i-th minibatch within total steps [start from 0]
        adap_training_steps: list of 3 cutoff index for different training stages [warmup, freeze, refine]
        return: stage string
    """
    if batch_idx < adap_training_steps[0]:
        stage = "warmup"
    elif batch_idx >= adap_training_steps[0] and batch_idx < adap_training_steps[1]:
        stage = "freeze"
    else:
        stage = "refine"

    return stage


def calculate_stage_cutoffs(total_steps, fractions):
    event_cutoffs = []
    cumulative_steps = 0
    for fraction in fractions:
        event_steps = int(total_steps * fraction)
        cumulative_steps += event_steps
        event_cutoffs.append(cumulative_steps)
    
    event_cutoffs[-1] = total_steps
    
    return event_cutoffs


def make_optimizer(params, optimizer_name, lr):
    if optimizer_name == "Adam":
        return torch.optim.Adam(params, lr)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(params, lr)

def make_scheduler(optimizer, scheduler_type, total_steps, warmup_steps=0):
    if scheduler_type == "constant":
        scheduler = transformers.get_constant_schedule(optimizer)
    elif scheduler_type == "constant_warmup":
        scheduler = transformers.get_constant_schedule_with_warmup(optimizer,
                        num_warmup_steps=warmup_steps) 
    elif scheduler_type == "cosine_warmup":
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
                                    num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    return scheduler

def make_model(config, vis):
    model = SwinAudioCodec(config.model.in_dim, config.model.in_freq, config.model.h_dims,
        config.model.swin_depth, config.model.swin_heads, config.model.window_size, 
        config.model.mlp_ratio, config.model.max_streams, config.model.overlap, 
        config.model.num_vqs, config.model.proj_ratio, config.model.codebook_size, config.model.codebook_dims,
        config.model.patch_size, config.model.use_ema, config.model.use_cosine_sim, 
        config.model.is_causal, config.model.vq, config.model.fuse_net, config.model.scalable, False,
        config.model.mel_windows, config.model.mel_bins, config.model.win_len,
        config.model.hop_len, config.model.sr, vis)
    return model

def make_metrics(device):
    return {
    "Test_PESQ": PESQ(sample_rate=16000, device=device),
    "Test_MelDist": MelDistance(win_lengths=[32,64,128,256,512,1024,2048], n_mels=[5,10,20,40,80,160,320]).to(device),
    "Test_STFTDist": STFTDistance(win_lengths=[2048,512]).to(device),
    "Test_PSNR": PSNR(), "Test_SNR":SNR()
    }

def switch_stage(model, stage="warmup->freeze", 
                 optimizer_name="AdamW", lr=1e-4):

    if stage == "warmup->freeze":
        model, trainable_params = freeze_layers(model, part="encoder")
        optimizer = make_optimizer(trainable_params, optimizer_name, lr)

    elif stage == "freeze->refine":
        model = unfreeze_layers(model, part="encoder")
        decay_lr = lr * .3
        optimizer = make_optimizer(model.parameters(), optimizer_name, decay_lr)

    return model, optimizer


if __name__ == "__main__":
    print(calculate_stage_cutoffs(total_steps=400000, fractions=[0.125,0.625,0.25])) # 10 50 20
    print(calculate_stage_cutoffs(total_steps=250000, fractions=[0.2,0.6,0.2])) # 10 30 10