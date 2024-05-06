import os, torchaudio
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

import argbind
import argparse, yaml
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms

from torch.utils.data import Dataset
from pesq import pesq, NoUtterancesError, BufferTooShortError

from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("..")

import dac
import wandb
from glob import glob

warnings.filterwarnings("ignore", category=UserWarning)

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="16kHz_dns_9k.yml")
    args = parser.parse_args()

    with open(f"../conf/{args.config}", "r") as f:
        args = yaml.safe_load(f)
    args = dict2namespace(args)
    return args

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform

def build_dataset(
    data_path: str = "/scratch/yg172/DNS_CHALLENGE/processed_wav",
    split: str = "train",
):  
    transform = build_transform()
    dataset = DNS(data_path, split)
    dataset.transform = transform
    return dataset

class DNS(Dataset):
    data_name = 'DNS'
    def __init__(self, data_path, split) -> None:
        self.split = split
        d_pth = '{}/{}'.format(data_path, split)

        self.source_audio = glob(f"{d_pth}/*/*.wav") # all wav paths
        self.source_audio = self.source_audio[:180000] if split == "train" else self.source_audio

    def __len__(self):
        return len(self.source_audio)

    def __getitem__(self, idx):
        x = {'signal': torchaudio.load(self.source_audio[idx])[0][:, :-80], } # [1*N]
        return x
    
def collate_fn(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys() if key != "feat"}  # output = {'audio':[.....], 'feature':[.....]}
        for b in batch:
            for key in b:
                output[key].append(b[key])

        for key in output:
            output[key] = AudioSignal(torch.stack(output[key], dim=0), sample_rate=16000)

        return output

def PESQ(recon: AudioSignal, raw: AudioSignal):
    score = 0.0
    for i in range(raw.batch_size):
        try:
            obj_score = pesq(16000, 
                            raw.audio_data[i].squeeze(0).cpu().numpy(), 
                            recon.audio_data[i].squeeze(0).cpu().numpy(), 'wb')
        except NoUtterancesError:
            obj_score = 0.0
        except BufferTooShortError:
            obj_score = 0.0
        
        score += obj_score
    
    return score/raw.batch_size
    
@dataclass
class State:
    generator: dac.model.DAC
    optimizer_g: torch.optim.AdamW
    scheduler_g: torch.optim.lr_scheduler.ExponentialLR

    stft_loss: dac.nn.loss.MultiScaleSTFTLoss
    mel_loss: dac.nn.loss.MelSpectrogramLoss
    waveform_loss: dac.nn.loss.L1Loss

    pesq_score: PESQ

    train_data: Dataset
    val_data: Dataset

    tracker: Tracker

def load(
    accel: ml.Accelerator,
    tracker: Tracker,
    args: argparse.Namespace
):

    # Models
    generator = dac.model.DAC(
        encoder_dim=args.DAC.encoder_dim,
        encoder_rates=args.DAC.encoder_rates,
        decoder_dim=args.DAC.decoder_dim,
        decoder_rates=args.DAC.decoder_rates,
        n_codebooks=args.DAC.n_codebooks,
        codebook_size=args.DAC.codebook_size,
        codebook_dim=args.DAC.codebook_dim,
        quantizer_dropout=args.DAC.quantizer_dropout,
        sample_rate=args.DAC.sample_rate
    )
    print("Generator num Params: ", sum(p.numel() for p in generator.parameters() if p.requires_grad))

    generator = accel.prepare_model(generator)

    optimizer_g = torch.optim.AdamW(generator.parameters(), 
                                    lr=args.AdamW.lr, betas=args.AdamW.betas,)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, args.ExponentialLR.gamma)

    print("Loading data...")
    train_data = build_dataset(data_path=args.data_path, split="train")
    val_data = build_dataset(data_path=args.data_path, split="test")
    train_data.collate = collate_fn
    val_data.collate = collate_fn
    print("data ready")

    waveform_loss = dac.nn.loss.L1Loss()
    stft_loss = dac.nn.loss.MultiScaleSTFTLoss(args.MultiScaleSTFTLoss.window_lengths)
    mel_loss = dac.nn.loss.MelSpectrogramLoss(args.MelSpectrogramLoss.n_mels,
                                              window_lengths=args.MelSpectrogramLoss.window_lengths,
                                              clamp_eps=args.MelSpectrogramLoss.clamp_eps,
                                              pow=args.MelSpectrogramLoss.pow,
                                              mag_weight=args.MelSpectrogramLoss.mag_weight,
                                              mel_fmin=args.MelSpectrogramLoss.mel_fmin,
                                              mel_fmax=args.MelSpectrogramLoss.mel_fmax,
                                              )

    eval_metric = PESQ
    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        pesq_score=eval_metric,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )


@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    for key in batch.keys():
        batch[key] = batch[key].to(accel.device)
    # batch = util.prepare_batch(batch, accel.device)
    # signal = state.val_data.transform(
    #     batch["signal"].clone(), #**batch["transform_args"]
    # )
    signal = batch["signal"].clone()

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    return {
        "loss": state.mel_loss(recons, signal),
        "mel/loss": state.mel_loss(recons, signal),
        "stft/loss": state.stft_loss(recons, signal),
        "waveform/loss": state.waveform_loss(recons, signal),
        "pesq": state.pesq_score(recons, signal)
    }

@timer()
def train_loop(state, batch, accel, lambdas):
    state.generator.train()
    output = {}

    # batch = util.prepare_batch(batch, accel.device)
    for key in batch.keys():
        batch[key] = batch[key].to(accel.device)
    with torch.no_grad():
        # signal = state.train_data.transform(
        #     batch["signal"].clone(), #**batch["transform_args"]
        # )
        signal = batch["signal"].clone()

    with accel.autocast():
        out = state.generator(signal.audio_data, signal.sample_rate)
        recons = AudioSignal(out["audio"], signal.sample_rate)
        commitment_loss = out["vq/commitment_loss"]
        codebook_loss = out["vq/codebook_loss"]

    with accel.autocast():
        output["stft/loss"] = state.stft_loss(recons, signal)
        output["mel/loss"] = state.mel_loss(recons, signal)
        output["waveform/loss"] = state.waveform_loss(recons, signal)
        output["vq/commitment_loss"] = commitment_loss
        output["vq/codebook_loss"] = codebook_loss
        output["loss"] = sum([v * output[k] for k, v in vars(lambdas).items() if k in output])

    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 1e3
    )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()

    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}


def validate(state, val_dataloader, accel):

    wandb_log = {}
    for batch in tqdm(val_dataloader, desc="Validating Model"):
        output = val_loop(batch, state, accel)
        for key, val in output.items():
            if not isinstance(val, float):
                val = val.item()
            key = f"test/{key}"
            if key not in wandb_log:
                wandb_log[key] = [val]
            else:
                wandb_log[key].append(val)

    for key in wandb_log.keys():
        wandb_log[key] = np.mean(wandb_log[key])

    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()

    return wandb_log

def checkpoint(state, step, score, best_score, 
               save_iters, save_path, accel):
    # metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    print(f"Saving to {str(Path('.').absolute())}")
    if score > best_score:
        print("Best generator so far")
        tags.append("best")
        best_score = score
    if step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            # "metadata.pth": metadata,
        }
        # accel.unwrap(state.generator).metadata = metadata
        accel.unwrap(state.generator).save_to_folder(
            f"{save_path}/{tag}", generator_extra
        )

    return best_score

def train_dns(
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    log_every: int = 5,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    lambdas: dict = {
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
    },
    args: argparse.Namespace=None
):
    if accel.local_rank == 0:
        wandb.login(key="880cb5a13d061af184bd6f3833bbce3df6d099fc")
        wandb.init(project=args.wb_project_name, name=args.wb_exp_name)

    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )
    state = load(accel, tracker, args)

    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=0 * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False,
    )

    print("Start Training...")
    # These functions run only on the 0-rank process
    # save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    global checkpoint
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)
    best_score = -1
    for step, batch in tqdm(enumerate(train_dataloader, start=1), desc="Training Model", total=num_iters):
        outputs = train_loop(state, batch, accel, lambdas)
        if step % log_every == 0:
            wandb_log_train = {f"train/{key}":val for key,val in outputs.items()}
            if wandb.run is not None and accel.local_rank == 0:
                wandb.log(wandb_log_train)

        last_iter = (
            step == num_iters - 1 if num_iters is not None else False
        )

        if step % valid_freq == 0 or last_iter:
            wandb_log_test = validate(state, val_dataloader, accel)
            if wandb.run is not None and accel.local_rank == 0:
                wandb.log(wandb_log_test)
            score = wandb_log_test["test/pesq"]
            best_score = checkpoint(state, step, score, best_score, save_iters, save_path, accel)

        if last_iter:
            break
    
    if accel.local_rank == 0:
        wandb.finish()

def main(args):
    print(f"Reproducing Experiment: DAC16kHz (without gan) on DNSCHALLENGE\nConfiguration:{args.wb_exp_name}")

    args.wb_exp_name += "_non_adversarial"
    with ml.Accelerator(amp=args.amp) as accel:
        if accel.local_rank != 0:
            sys.tracebacklimit = 0
        train_dns(accel, args.seed, args.save_path, args.num_iters,
                  args.save_iters, args.sample_freq, args.log_every, args.valid_freq,
                  args.batch_size, args.val_batch_size, args.num_workers,
                  args.val_idx, args.lambdas, args)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
    

"""
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node gpu train_customize.py --config 16kHz_dns_9k.yml

export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node gpu train_customize.py --config 16kHz_dns_9k_tiny.yml
"""
