import numpy as np
import torch, transformers
import torch.nn.functional as F
import sys
sys.path.append("../")

from models.codec import SwinAudioCodec, CrossScaleProgressiveResCodec
from models.residual_csvq import ResidualCrossScaleCodec
from models.losses.metrics import PESQ, MelDistance, SISDRLoss, STFTDistance, PSNR, SNR
from models.vq.quantization import GroupVQ

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
    elif scheduler_type == "exponential_decay":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999996)

    return scheduler

def make_model(config, vis, model="swin_codec"):
    if model == "swin_codec":
        model = SwinAudioCodec(config.model.in_dim, config.model.in_freq, config.model.h_dims,
            config.model.swin_depth, config.model.swin_heads, config.model.window_size, 
            config.model.mlp_ratio, config.model.max_streams, config.model.overlap, 
            config.model.num_vqs, config.model.proj_ratio, config.model.codebook_size, config.model.codebook_dims,
            config.model.patch_size, config.model.use_ema, config.model.use_cosine_sim, config.model.vq, config.model.kmeans_init,
            config.model.scalable, config.model.mel_windows, config.model.mel_bins, config.model.win_len,
            config.model.hop_len, config.model.sr, vis)
    elif model == "progressive_codec":
        model = CrossScaleProgressiveResCodec(config.model.in_dim, config.model.in_freq, config.model.h_dims,
            config.model.swin_depth, config.model.swin_heads, config.model.window_size, 
            config.model.mlp_ratio, config.model.max_streams, config.model.overlap, 
            config.model.num_vqs, config.model.proj_ratio, config.model.codebook_size, config.model.codebook_dims,
            config.model.patch_size, config.model.use_ema, config.model.use_cosine_sim, config.model.vq, config.model.scalable,
            config.model.mel_windows, config.model.mel_bins, config.model.win_len,
            config.model.hop_len, config.model.sr, vis)
    elif model == "residual_csvq_codec":
        model = ResidualCrossScaleCodec(config.model.in_dim, config.model.in_freq, config.model.h_dims, 
                 config.model.max_streams, 
                 config.model.overlap, config.model.num_vqs, config.model.proj_ratio,
                 config.model.codebook_size, config.model.codebook_dims, config.model.use_ema, config.model.use_cosine_sim,
                 config.model.patch_size, config.model.swin_depth, config.model.swin_heads,
                 config.model.window_size, config.model.mlp_ratio,
                 config.model.mel_windows, config.model.mel_bins,
                 config.model.win_len, config.model.hop_len, config.model.sr, config.model.vq, vis)
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

def reset_codebooks(gvq):
    """ set groupvq initialized flag to false, for initialize again """
    assert isinstance(gvq, GroupVQ), "Specified Module is not GroupVQ"
    gvq.codebook_initialized.fill_(0)

class Entropy_Counter:
    def __init__(self, codebook_size=1024, num_streams=6, num_groups=3, device="cuda"):

        self.vq_distributions = {
                f"stream_{S}_group_{G+1}": torch.zeros(codebook_size, device=device) for S in range(num_streams) for G in range(num_groups)
            }
        self.counts = 0
        
        self.num_streams = num_streams
        self.num_groups = num_groups
        self.codebook_size = codebook_size
        self.device = device

        self.max_entropy_per_book = np.log2(codebook_size)
        self.max_total_entropy = len(self.vq_distributions) * self.max_entropy_per_book

        self.entropy = None

    def update(self, multi_codes):
        """
        Args:
            multi_codes: [bs,G,T,num_streams] (T=500 in test T=150 in train)
            return_all_scale: return bitrate efficiency
        """ 
        assert multi_codes.size(-1) == self.num_streams and multi_codes.size(1) == self.num_groups, "code indices size not match"
        num_of_code = multi_codes.size(0) * multi_codes.size(2)

        for s in range(self.num_streams):
            stream_s_code = multi_codes[:, :, :, s] # bs, G, T
            for g in range(self.num_groups):
                stream_s_group_g_code = stream_s_code[:,g,:] # bs, T
                one_hot = F.one_hot(stream_s_group_g_code, num_classes=self.codebook_size) # bs, T, codebook_size
                self.vq_distributions[f"stream_{s}_group_{g+1}"] += one_hot.view(-1, self.codebook_size).sum(0) # (bs*T, codebook_size)
        self.counts += num_of_code # bs*T

    def compute_distribution(self,):
        for k, _counts in self.vq_distributions.items():
            self.vq_distributions[k] = _counts / torch.tensor(self.counts, device=_counts.device)
        return self.vq_distributions
    
    def compute_entropy(self, return_total=False):
        self.compute_distribution()
        entropy = {}
        for k, dist in self.vq_distributions.items():
            entropy[k] = (- torch.sum(dist * torch.log2(dist + 1e-10))).item()
    
        if return_total:
            return sum(entropy.values())
        
        return entropy
    
    def compute_bitrate_efficiency(self, return_total=False):
        if self.entropy is None:
            self.entropy = self.compute_entropy(False)

        efficiency = {}
        for k, e in self.entropy.items():
            efficiency[k] = round(e / self.max_entropy_per_book, 4)

        if return_total:
            return sum(self.entropy.values()) / self.max_total_entropy
        
        return efficiency

    def reset(self):
        self.vq_distributions = {
                f"stream_{S}_group_{G+1}": torch.zeros(self.codebook_size, device=self.device) \
                    for S in range(self.num_streams) for G in range(self.num_groups)
            }
        self.counts = 0
        self.entropy = None






if __name__ == "__main__":
    print(calculate_stage_cutoffs(total_steps=400000, fractions=[0.125,0.625,0.25])) # 10 50 20
    print(calculate_stage_cutoffs(total_steps=250000, fractions=[0.2,0.6,0.2])) # 10 30 10