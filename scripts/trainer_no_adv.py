import os, torch, wandb, random, argparse
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator

from models import make_model
from scripts.test import eval_epoch
from scripts.metrics import PESQ, MelSpectrogramDistance, SISDR, EntropyCounter
from scripts.utils import *

class Trainer:
    """Distributed Codec Trainer (non-adversarial)"""
    def __init__(self, accel: Accelerator, config: argparse.Namespace, args: argparse.Namespace) -> None:
        
        self.accel, self.config, self.args = accel, config, args
        self.log_stats = None

    def load(self):
        # Model
        model = make_model(self.config.model, self.config.model_name)
        n_params = sum(p.numel() for p in model.parameters())
        
        # Metrics and Losses
        self.metrics = {"PESQ": PESQ(), "MelDistance": MelSpectrogramDistance().to(self.accel.device), "SISDR": SISDR().to(self.accel.device)}
        self.e_counter = EntropyCounter(self.config.model.codebook_size, 
                                   self.config.model.max_streams if "csvq" in self.config.model_name else self.config.model.num_rvqs, 
                                   self.config.model.group_size if "csvq" in self.config.model_name else 1, device=self.accel.device)
        self.loss_funcs = {"mel_loss": make_losses(name="mel_loss").to(self.accel.device),
                           "stft_loss": make_losses(name="stft_loss").to(self.accel.device),}
        
        # DataLoaders
        train_dl, val_dl = make_dataloader(self.config.data.train_data_path, self.config.data.train_bs_per_device, 
                                           True, self.config.data.num_workers), \
                           make_dataloader(self.config.data.val_data_path, self.config.data.val_bs_per_device, 
                                           False, self.config.data.num_workers)  
        self.args.train_steps, test_steps = len(train_dl)//self.args.num_devices, len(val_dl)//self.args.num_devices
        self.args.max_train_steps = self.args.train_steps*self.args.num_epochs
        self.args.pretraining_steps = self.args.train_steps*self.args.num_pretraining_epochs
        
        # Optimizers
        optimizer = make_optimizer(model.parameters(), self.args.lr)
        scheduler = make_scheduler(optimizer, self.args.scheduler_type, 
                                   total_steps=self.args.max_train_steps, 
                                   warmup_steps=self.args.num_warmup_steps)
        
        self.accel.print(f"<<<<Experimental Setup: {self.args.exp_name}>>>>")
        self.accel.print(f"   BatchSize_per_Device: Train {self.config.data.train_bs_per_device} Test {self.config.data.val_bs_per_device}    LearningRate: {self.args.lr}")
        self.accel.print(f"   Total_Training_Steps: {self.args.train_steps}*{self.args.num_epochs}={self.args.max_train_steps}")
        self.accel.print(f"   Pre-Training_Steps: {self.args.train_steps}*{self.args.num_pretraining_epochs}={self.args.pretraining_steps}")
        self.accel.print(f"   Optimizer: AdamW    Scheduler: {self.args.scheduler_type}")
        self.accel.print(f"   Quantization_Dropout: {self.args.dropout_rate}")
        self.accel.print(f"   Model #Parameters: {n_params/1000000:.2f}M")

        self.bps_per_stream = 1.5 if "csvq" in self.config.model_name else 0.5

        return model, optimizer, scheduler, train_dl, val_dl
    
    def train(self, ):
        model, optimizer, scheduler, train_dl, val_dl = self.load()
        self.train_dl, self.val_dl = self.accel.prepare(train_dl), val_dl # No Distributing on Valset
        
        if self.args.pretrain_ckp is not None:
            ckp = torch.load(self.args.pretrain_ckp,)
            model.load_state_dict(ckp["model_state_dict"])
            optimizer.load_state_dict(ckp["optimizer_state_dict"])
            scheduler.load_state_dict(ckp["scheduler_state_dict"])
            self.start_step, self.best_perf = ckp["step"]+1, ckp['best_perf'] 
            self.accel.print(f"Load Pretrained Encoder-Decoder Checkpoints\nPrevious Best Performance: {self.best_perf} Starting Step: {self.start_step}")
        else:
            self.start_step, self.best_perf = 0, -1 
        
        self.model, self.optimizer, self.scheduler = self.accel.prepare(model, optimizer, scheduler) 
        self.pbar = tqdm(initial=self.start_step, total=self.args.max_train_steps, position=0, leave=True)
        while True:
            for _, x in enumerate(self.train_dl):
                self.train_step(x)
                if self.accel.is_main_process:
                    if self.pbar.n > self.args.pretraining_steps and (self.pbar.n+1) % self.args.train_steps==0:
                        self.evaluate()
                    if (self.pbar.n+1) % self.args.log_steps==0:
                        self.log_step()
                self.accel.wait_for_everyone()

                if self.pbar.n == self.args.pretraining_steps and self.pbar.n > 0:
                    if self.accel.is_main_process:
                        self.save_ckp(save_pth=f"{self.args.save_path}/{self.args.exp_name}",tag="pretrained.pth")
                    
                    # start training involving vqs: initialization
                    if "csvq" in self.config.model_name: 
                        for pvq in self.accel.unwrap_model(self.model).quantizers:
                            pvq.verbose_init = self.accel.is_main_process
                            pvq.codebook_initialized.fill_(0)
                    elif "rvq" in self.config.model_name:
                        for vq in self.accel.unwrap_model(self.model).quantizers.vqs:
                            torch.nn.init.kaiming_normal_(vq.embedding.weight) 	

                self.accel.wait_for_everyone()
                
                self.pbar.update(1)
                if self.pbar.n == self.args.max_train_steps: return 

    def train_step(self, x):
        
        # VQ Dropout and Pre-Training
        s = quantization_dropout(dropout_rate=self.args.dropout_rate, 
                max_streams=self.config.model.max_streams if "csvq" in self.config.model_name else self.config.model.num_rvqs)
        freeze_vq = self.pbar.n < self.args.pretraining_steps
        
        stage = "Pre-Training at 0kbps" if freeze_vq else f"Sampling at {s*self.bps_per_stream:.2f}kbps"
        self.pbar.set_description(f"Training Model [{stage}]")

        # Forward Pass
        outputs = self.model(**dict(x=x, x_feat=None, num_streams=s, freeze_codebook=freeze_vq))
        outputs["mel_loss"] = self.loss_funcs["mel_loss"](outputs["raw_audio"], outputs["recon_audio"])
        outputs["stft_loss"] = self.loss_funcs["stft_loss"](outputs["raw_feat"], outputs["recon_feat"])
        outputs["loss"] = outputs["cm_loss"]*self.config.loss.cm_weight + \
                          outputs["cb_loss"]*self.config.loss.cb_weight + \
                          outputs["mel_loss"]*self.config.loss.mel_weight + \
                          outputs["stft_loss"]*self.config.loss.stft_weight
        
        # Backward Pass
        self.optimizer.zero_grad()
        self.accel.backward(outputs["loss"].mean())
        self.accel.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        self.scheduler.step()
        
        # Store Logs
        if self.log_stats is None:
            self.log_stats = {k: [] for k in outputs.keys() if k.split("_")[-1] == "loss"}
        for k in self.log_stats.keys():
            self.log_stats[k].append(outputs[k].mean().item())

    def log_step(self):
        for k, v in self.log_stats.items():
            self.log_stats[k] = np.mean(v)
        if wandb.run is not None: wandb.log(self.log_stats)
        self.log_stats = None
    
    def evaluate(self, ):
        # Validation Epoch
        eval_streams = self.config.model.max_streams if "csvq" in self.config.model_name else self.config.model.num_rvqs
        perf = eval_epoch(model=self.accel.unwrap_model(self.model).to(self.accel.device), 
                          eval_loader=self.val_dl, metric_funcs=self.metrics, e_counter=self.e_counter,
                          device=self.accel.device, num_streams=eval_streams, verbose=False)

        # wandb logging
        perf = {k:v[0] for k,v in perf.items()}
        if wandb.run is not None: wandb.log(perf)
        self.accel.print(f"[Step {self.pbar.n+1}/{self.args.max_train_steps}] | Performance at {eval_streams*self.bps_per_stream:.2f}kbps:\n", 
                         " | ".join(f"{k}: {v:.4f}" for k, v in perf.items()))

        # Saving Checkpoints
        if perf[self.args.val_metric] > self.best_perf: 
            self.best_perf = perf[self.args.val_metric]
            self.save_ckp(save_pth=f"{self.args.save_path}/{self.args.exp_name}",tag="best.pth")
        self.save_ckp(save_pth=f"{self.args.save_path}/{self.args.exp_name}",tag="checkpoint.pth")

    def save_ckp(self, save_pth, tag="file.pth"):
        ckp = {
            'step': self.pbar.n, 
            'model_state_dict': self.accel.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.accel.unwrap_model(self.optimizer).state_dict(), 
            'scheduler_state_dict': self.accel.unwrap_model(self.scheduler).state_dict(),
            "best_perf": self.best_perf
        }
        if not os.path.exists(save_pth): os.makedirs(save_pth)
        self.accel.save(ckp, os.path.join(save_pth,tag))
        self.accel.print(f"[Step {self.pbar.n+1}/{self.args.max_train_steps}] | Training checkpoint saved as {os.path.join(save_pth,tag)}")


def main(args, config):
    accel = Accelerator()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if accel.is_main_process:
        if args.wandb_project is not None:
            wandb.login()
            wandb.init(project=args.wandb_project, name=args.exp_name)
        else:   
            accel.print("Deactivated WandB Logging")

    trainer = Trainer(accel, config, args)
    trainer.train()
    if accel.is_main_process:
        wandb.finish()