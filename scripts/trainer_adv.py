import os, random, wandb
from argparse import Namespace
from accelerate import Accelerator
from tqdm import tqdm
from .trainer_no_adv import Trainer

from esc.models import Discriminator, make_model
from esc.modules import GANLoss
from .metrics import PESQ, MelSpectrogramDistance, SISDR, EntropyCounter
from .utils import *

class TrainerAdv(Trainer):
    def __init__(self, accel: Accelerator, config: Namespace, args: Namespace) -> None:
        super().__init__(accel, config, args)

    def load(self):
        # Model
        model_gen = make_model(self.config.model, self.config.model_name)
        n_params_gen = sum(p.numel() for p in model_gen.parameters())

        model_disc = Discriminator(**namespace2dict(self.config.discriminator))
        n_params_disc = sum(p.numel() for p in model_disc.parameters())
        
        # Metrics and Losses
        self.metrics = {"PESQ": PESQ(), "MelDistance": MelSpectrogramDistance().to(self.accel.device), "SISDR": SISDR().to(self.accel.device)}
        self.e_counter = EntropyCounter(self.config.model.codebook_size, self.config.model.max_streams, device=self.accel.device)
        self.loss_funcs = {"mel_loss": make_losses(name="mel_loss").to(self.accel.device),
                           "stft_loss": make_losses(name="stft_loss").to(self.accel.device)}
        
        # DataLoaders
        train_dl, val_dl = make_dataloader(self.config.data.train_data_path, self.config.data.train_bs_per_device, 
                                           True, self.config.data.num_workers), \
                           make_dataloader(self.config.data.val_data_path, self.config.data.val_bs_per_device, 
                                           False, self.config.data.num_workers)  
        self.args.train_steps, test_steps = len(train_dl)//self.args.num_devices, len(val_dl)//self.args.num_devices
        self.args.max_train_steps = self.args.train_steps*self.args.num_epochs
        self.args.pretraining_steps = self.args.train_steps*self.args.num_pretraining_epochs
        
        # Optimizers
        self.args.lr_disc = self.args.lr
        if self.args.pretrain_ckp is not None: self.args.lr = self.args.lr / 10
        optimizer_gen = make_optimizer(model_gen.parameters(), self.args.lr)
        scheduler = make_scheduler(optimizer_gen, self.args.scheduler_type, 
                                   total_steps=self.args.max_train_steps, 
                                   warmup_steps=self.args.num_warmup_steps)
        
        optimizer_disc = make_optimizer(model_disc.parameters(), self.args.lr_disc)
        
        self.accel.print(f"<<<<Experimental Setup: {self.args.exp_name}>>>>")
        self.accel.print(f"   BatchSize_per_Device: Train {self.config.data.train_bs_per_device} Test {self.config.data.val_bs_per_device}\n   LearningRate(gen): {self.args.lr}    LearningRate(disc): {self.args.lr_disc}")
        self.accel.print(f"   Total_Training_Steps: {self.args.train_steps}*{self.args.num_epochs}={self.args.max_train_steps}")
        self.accel.print(f"   Pre-Training_Steps: {self.args.train_steps}*{self.args.num_pretraining_epochs}={self.args.pretraining_steps}")
        self.accel.print(f"   Optimizer: AdamW    Scheduler: {self.args.scheduler_type}")
        self.accel.print(f"   Quantization_Dropout: {self.args.dropout_rate}")
        self.accel.print(f"   Model #Parameters: {n_params_gen/1e6:.2f}M  Discriminator #Parameters: {n_params_disc/1e6:.2f}M")

        self.bps_per_stream = 1.5

        return model_gen, model_disc, optimizer_gen, optimizer_disc, scheduler, train_dl, val_dl
    
    def train_step(self, x):
        
        # VQ Dropout and Pre-Training
        s = quantization_dropout(dropout_rate=self.args.dropout_rate, max_streams=self.config.model.max_streams)
        freeze_vq = self.pbar.n < self.args.pretraining_steps
        
        stage = "Pre-Training at 0kbps" if freeze_vq else f"Sampling at {s*self.bps_per_stream:.2f}kbps"
        self.pbar.set_description(f"Training Model [{stage}]")

        # Forward Pass (Generator)
        outputs = self.model(**dict(x=x, x_feat=None, num_streams=s, freeze_codebook=freeze_vq))
        outputs["mel_loss"] = self.loss_funcs["mel_loss"](outputs["raw_audio"], outputs["recon_audio"])
        outputs["stft_loss"] = self.loss_funcs["stft_loss"](outputs["raw_feat"], outputs["recon_feat"])
        
        if not freeze_vq:
            outputs["gen_loss"], outputs["feat_loss"] = self.loss_funcs["adv_loss"].generator_loss(
                fake=outputs["recon_audio"], real=outputs["raw_audio"]
            )
        else:
            outputs["gen_loss"], outputs["feat_loss"] = torch.zeros(x.size(0), device=x.device), torch.zeros(x.size(0), device=x.device)
        
        outputs["loss"] = outputs["cm_loss"]*self.config.loss.cm_weight + \
                          outputs["cb_loss"]*self.config.loss.cb_weight + \
                          outputs["mel_loss"]*self.config.loss.mel_weight + \
                          outputs["stft_loss"]*self.config.loss.stft_weight + \
                          outputs["gen_loss"]*self.config.loss.gen_weight + \
                          outputs["feat_loss"]*self.config.loss.feat_weight
        
        # Backward Pass (Generator)
        self.opt_g.zero_grad()
        self.accel.backward(outputs["loss"].mean())
        self.accel.clip_grad_norm_(self.model.parameters(), 1e3)
        self.opt_g.step()
        self.scheduler.step()

        if not freeze_vq: # discriminator involved only after pre-training
            # Forward Pass (Discriminator)
            outputs["disc_loss"] = self.loss_funcs["adv_loss"].discriminator_loss(
                fake=outputs["recon_audio"], real=outputs["raw_audio"]
            )
            # Backward Pass (Discriminator)
            self.opt_d.zero_grad()
            self.accel.backward(outputs["disc_loss"].mean())
            self.accel.clip_grad_norm_(self.model_disc.parameters(), 10.0)
            self.opt_d.step()
        else:
            outputs["disc_loss"] = torch.zeros(x.size(0), device=x.device)

        # Store Logs
        if self.log_stats is None:
            self.log_stats = {k: [] for k in outputs.keys() if k.split("_")[-1] == "loss"}
        for k in self.log_stats.keys():
            self.log_stats[k].append(outputs[k].mean().item())

    def train(self, ):
        g, d, opt_g, opt_d, scheduler, train_dl, val_dl = self.load()
        self.train_dl, self.val_dl = self.accel.prepare(train_dl), val_dl # No Distributing on Valset
        
        if self.args.pretrain_ckp is not None: # when provided, means post adversarial training instead of resume
            ckp = torch.load(self.args.pretrain_ckp, map_location='cpu')
            g.load_state_dict(ckp["model_state_dict"])
            if 'optimizer_state_dict' in ckp:
                opt_g.load_state_dict(ckp["optimizer_state_dict"])
            if 'scheduler_state_dict' in ckp:
                scheduler.load_state_dict(ckp["scheduler_state_dict"])
            self.accel.print(f"Load a Pretrained ESC Codec Generator\n---Start Post Adversarial Training---")

        self.start_step, self.best_perf = 0, -1
        self.pbar = tqdm(initial=self.start_step, total=self.args.max_train_steps, position=0, leave=True)
        self.model, self.model_disc, self.opt_g, self.opt_d, self.scheduler = self.accel.prepare(g, d, opt_g, opt_d, scheduler) 
        self.loss_funcs["adv_loss"] = GANLoss(self.model_disc).to(self.accel.device)

        if self.args.pretrain_ckp is not None and self.accel.is_main_process: 
            self.evaluate() # pre-eval epoch 
        self.accel.wait_for_everyone()       
        
        while True:
            for _, x in enumerate(self.train_dl):

                if self.args.pretraining_steps > 0 and self.pbar.n == self.args.pretraining_steps+1:
                    opt_g = make_optimizer(self.accel.unwrap_model(self.model).parameters(), self.args.lr)
                    self.opt_g = self.accel.prepare(opt_g)
                    self.accel.print("Pretraining done. Generator's Optimizer Renewed")

                self.train_step(x)
                
                if self.accel.is_main_process:
                    if self.pbar.n > self.args.pretraining_steps and self.pbar.n % self.args.train_steps==0:
                        self.evaluate()
                        self.model.train()
                    if (self.pbar.n+1) % self.args.log_steps==0:
                        self.log_step()
                    if self.pbar.n == self.args.pretraining_steps and self.pbar.n > 0:
                        self.save_ckp(save_pth=f"{self.args.save_path}/{self.args.exp_name}",tag="pretrained.pth")
                self.accel.wait_for_everyone()
                	
                self.pbar.update(1)
                if self.pbar.n == self.args.max_train_steps: return 

    def save_ckp(self, save_pth, tag="file.pth"):
        ckp = {
            'step': self.pbar.n, 
            'model_state_dict': self.accel.unwrap_model(self.model).state_dict(),
            'model_disc_state_dict': self.accel.unwrap_model(self.model_disc).state_dict(),
            'optimizer_state_dict': self.accel.unwrap_model(self.opt_g).state_dict(), 
            'optimizer_disc_state_dict': self.accel.unwrap_model(self.opt_d).state_dict(),
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

    trainer_adv = TrainerAdv(accel, config, args)
    trainer_adv.train()
    if accel.is_main_process:
        wandb.finish()