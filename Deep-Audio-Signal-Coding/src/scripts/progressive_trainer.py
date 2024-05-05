import os, torch, wandb, random, torchaudio

import sys
sys.path.append("/Users/tracy/Library/CloudStorage/GoogleDrive-cloudstorage.yuzhe@gmail.com/My Drive/Research/Audio_Signal_Coding/Deep-Audio-Signal-Coding/src")

from data import make_data_loader, fetch_dataset
from scripts.utils import quantization_dropout, make_optimizer, make_scheduler, make_model, make_metrics

import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from accelerate import Accelerator
import matplotlib.pyplot as plt

class ProgressiveTrainer:
    def __init__(self, accel, config, args) -> None:
        
        self.args = args
        self.config = config
        self.accel = accel
        
        self.evaluation = None
        self.best_perf = None
        self.progress_bar = None
        self.stage = "stabilize_step_0"

        if not os.path.exists(f"{args.save_dir}/{args.wb_exp_name}"):
            os.makedirs(f"{args.save_dir}/{args.wb_exp_name}")
        self.accel.print(f"Saving outputs into {args.save_dir}/{args.wb_exp_name}")
    
    def _train_one_step(self, input):
        self.check_progressive_stage(self.progress_bar.n)
        if self.stage == "dropout_step":
            streams = quantization_dropout(self.args.q_dropout_rate, self.config.model.max_streams)
            self.progress_bar.set_description(f"Training Model [{self.stage}][dropout sampling]")
        else:
            streams = int(self.stage.split("_")[-1]) + 1
            self.progress_bar.set_description(f"Training Model [{self.stage}][{(streams*1.5):.2f}kbps]")

        alpha = self.compute_fade_in_alpha(self.progress_bar.n)
        output = self.generator(**dict(x=input["audio"], x_feat=input["feat"] if "feat" in input else None, 
                        alpha=alpha, streams=streams, train=True))

        # Update Generator
        output["loss"] = self.config.loss.recon_factor * output["recon_loss"] + \
                            self.config.loss.commitment_factor * output["commitment_loss"] + \
                                self.config.loss.codebook_factor * output["codebook_loss"] + \
                                    self.config.loss.mel_factor * output["mel_loss"] + \
                                        self.config.loss.reg_factor * output["kl_loss"]
        self.optimizer.zero_grad()
        self.accel.backward(output["loss"].mean())
        self.accel.clip_grad_norm_(self.generator.parameters(), 0.5)
        self.optimizer.step()
        self.scheduler.step()

        self.progress_bar.update(1)
        # Logging batch losses
        if self.accel.is_main_process:
            if self.evaluation is None:
                self.evaluation = {k: [] for k in output.keys() if k in ["loss", "recon_loss", 
                                    "commitment_loss", "codebook_loss", "mel_loss", "kl_loss",]}
                self.evaluation["alpha"] = []
            for key, _ in self.evaluation.items():
                if key == "alpha":
                    self.evaluation[key].append(alpha) 
                else:
                    self.evaluation[key].append(output[key].mean().item())

            if self.progress_bar.n in self.args.save_steps:
                os.makedirs(f"{self.args.save_dir}/{self.args.wb_exp_name}/train_log/", exist_ok=True)
                torchaudio.save(f"{self.args.save_dir}/{self.args.wb_exp_name}/train_log/raw_audio_step_{self.progress_bar.n}.wav",
                                input["audio"][0:1].cpu(), 16000)
                torchaudio.save(f"{self.args.save_dir}/{self.args.wb_exp_name}/train_log/recon_audio_step_{self.progress_bar.n}.wav",
                                output["recon_audio"][0:1].cpu(), 16000)
    
    def _log_train_batch(self):
        for key, val in self.evaluation.items():
            self.evaluation[key] = np.mean(val)
        if wandb.run is not None:
            wandb.log(self.evaluation)
        self.evaluation = None

    def _test_batch(self, input, streams):
        self.generator.eval() 
        output = self.generator(**dict(x=input["audio"], 
                            x_feat=input["feat"] if "feat" in input else None, 
                            streams=streams, train=False))
        local_obj_metric = {} # metric stats on each device
        for k, m in self.obj_metric.items():
            if k in ['Test_PESQ', 'Test_MelDist', 'Test_STFTDist', 'Test_SNR']:
                local_obj_metric[k] = m(input["audio"], output["recon_audio"]) # Tensor
            elif k in ["Test_PSNR"]:
                local_obj_metric[k] = m(output["raw_feat"], output["recon_feat"]) # Tensor

        local_obj_metric_gathered = {} # metric stats gathered from all device
        for k, v in local_obj_metric.items():
            local_obj_metric_gathered[k] = self.accel.gather(v.to(self.accel.device)).tolist()

        for k, v_g in local_obj_metric_gathered.items(): # cummulate for all batches
            self.cumm_metric[k].extend(v_g)

    def _test_epoch(self, streams=None):
        """Distributed Evaluate"""
        self.cumm_metric = {k:[] for k in self.obj_metric.keys()}
        if streams is None: 
            streams = int(self.stage.split("_")[-1]) + 1 if self.stage != "dropout_step" else self.config.model.max_streams
        if self.accel.is_main_process:
            for _, input in tqdm(enumerate(self.test_data), total=len(self.test_data), 
                                desc=f"Eval at step {self.progress_bar.n+1}/{self.total_training_steps}"): 
                self._test_batch(input, streams)
        else:
            for _, input in enumerate(self.test_data): 
                self._test_batch(input, streams)

        if self.accel.is_main_process:
            test_performance = self._log_test_batch()
            self.accel.print(f"{self.stage}[{(1.5*streams):.2f}kbps] | ", " | ".join(f"{k}: {v:.4f}" for k, v in test_performance.items()))
            if self.stage == "dropout_step": # save best in dropout_step
                if test_performance["Test_MelDist"] < self.best_perf:
                    self.best_perf = test_performance["Test_MelDist"]
                    self.accel.print(f"Found Best Model at step {self.progress_bar.n+1}")
                    os.makedirs(f"{self.args.save_dir}/{self.args.wb_exp_name}/{self.stage}", exist_ok=True)
                    self._save_checkpoint(self.progress_bar.n+1, save_pth=f"{self.args.save_dir}/{self.args.wb_exp_name}/{self.stage}/best.pt")
            if self.stage.split("_")[0] == "stabilize": # save checkpoint in stabilize_step
                self.accel.print(f"Saving Checkpoint at step {self.progress_bar.n+1}")
                if not os.path.exists(f"{self.args.save_dir}/{self.args.wb_exp_name}/{self.stage}"):
                    os.makedirs(f"{self.args.save_dir}/{self.args.wb_exp_name}/{self.stage}")
                self._save_checkpoint(self.progress_bar.n+1, save_pth=f"{self.args.save_dir}/{self.args.wb_exp_name}/{self.stage}/checkpoint.pt")
        
        self.accel.wait_for_everyone()

    def _log_test_batch(self):
        test_performance = {}
        for k, v in self.cumm_metric.items():
            test_performance[k] = np.mean(v)
        if wandb.run is not None:
            wandb.log(test_performance)
        return test_performance 

    def train(self, ):
        # Prepare steps for progressive training
        self.progressive_stages, self.progressive_cutoffs, self.total_training_steps = compute_full_cutoffs(
            self.config.training.warmup_steps, self.config.training.progressive_steps, self.config.training.stabilize_steps, self.config.training.dropout_steps)
        idx = self.progressive_stages.index(self.stage)
        self.current_cutoff = self.progressive_cutoffs[idx]
        self.next_stage = self.progressive_stages[idx+1]
        self.next_cutoff = self.progressive_cutoffs[idx+1]
        
        # Prepare Dataloaders
        dls = self.prepare_dataloader(self.args, self.config)
        self.train_data = self.accel.prepare(dls["train"])
        self.test_data = self.accel.prepare(dls["test"])
        # Prepare Model Optimizer Scheduler Metrics 
        generator, optimizer, self.scheduler, self.obj_metric = self.load_train_objs(self.config, self.args)
        self.generator, self.optimizer = self.accel.prepare(generator, optimizer)

        self.accel.print(f"Running Experiment: {self.args.wb_exp_name}")
        self.accel.print(f"learning_rate: {self.args.lr} | batch_size: {self.args.train_bs_per_device * self.args.num_device} | num_worker: {self.args.num_worker}")
        self.accel.print("Progressive Training Steps: ", self.config.training.progressive_steps, 
                         "\nStabilize Training Steps: ", self.config.training.stabilize_steps, 
                         "\nDropout Training Steps: ", self.config.training.dropout_steps, " Warmup Training Steps: ", self.config.training.warmup_steps)
        
        # Initialize from a pretrained autoencoder
        if self.args.init_ckpt is not None: 
            if os.path.exists(self.args.init_ckpt):
                self.accel.print(f"Initialize Encoder-Decoder from {self.args.init_ckpt}")
                self._load_checkpoint(self.args.init_ckpt, model_only=True)
        
        self.start_step, self.best_perf = 0, np.inf
        self.progress_bar = tqdm(initial=self.start_step, total=self.total_training_steps)
        while True:
            flag=False
            for _, input in enumerate(self.train_data):
            
                self._train_one_step(input)

                if (self.progress_bar.n+1) % self.args.info_steps == 0 and self.accel.is_main_process:
                    self._log_train_batch()

                if (self.progress_bar.n+1) % self.args.eval_every == 0:
                    self._test_epoch()
                    # if self.stage not in ["dropout_step", "stabilize_step_0"]: # evaluate previous bitstream also (catastrophic forgetting)
                    #     progressive_step = int(self.stage.split("_")[-1])
                    #     self.accel.print("Evaluate Previous one Bitstream Also...")
                    #     self._test_epoch(streams=progressive_step)

                if self.progress_bar.n == 250000 and self.accel.is_main_process: # for comparing with baselines
                    self._save_checkpoint(self.progress_bar.n, 
                        save_pth=f"{self.args.save_dir}/{self.args.wb_exp_name}/250ksteps_ckp.pt")

                if self.progress_bar.n == self.total_training_steps:
                    flag = True
                    break

            if flag: break

    def compute_fade_in_alpha(self, step: int):
        """compute hyperparameter alpha in progressive training
        Args:   
            step: current num of updated steps in total (self.progress_bar.n)
            returns: blending parameter alpha
        """
        if self.stage.split("_")[0] == "progressive":
            alpha = 1.0 * (step-self.current_cutoff) / (self.next_cutoff-self.current_cutoff)
        else:
            alpha = 1.0
        return alpha

    def check_progressive_stage(self, step: int):
        """check current progressive training stage and update
        Args:   
            step: current num of updated steps in total (self.progress_bar.n)
        """
        if step == self.next_cutoff: # update stage
            next_stage = self.next_stage.split("_")[0]
            # if next_stage in ["progressive", "dropout"]:
            #     if next_stage == "dropout": current_progressive_step = 6
            #     else: current_progressive_step = int(self.next_stage.split("_")[-1])
            #     self._save_checkpoint(self.progress_bar.n, 
            #         save_pth=f"{self.args.save_dir}/{self.args.wb_exp_name}/{current_progressive_step*1.5}kbps_progressive_step{current_progressive_step}_step{self.progress_bar.n}_ckp.pt")
            self.stage = self.next_stage
            idx = self.progressive_stages.index(self.stage)
            self.current_cutoff = self.progressive_cutoffs[idx]
            if self.stage != "dropout_step":
                self.next_stage = self.progressive_stages[idx+1]
                self.next_cutoff = self.progressive_cutoffs[idx+1]

    def _save_checkpoint(self, step, save_pth):
        """Save accel.prepare(object) checkpoints"""
        # self.accel.wait_for_everyone()
        ckp = { 'step': step, 
                'model_state_dict': self.accel.unwrap_model(self.generator).state_dict(),
                'optimizer_state_dict': self.accel.unwrap_model(self.optimizer).state_dict(), 
                'scheduler_state_dict': self.scheduler.state_dict(),
                "best_perf": self.best_perf, 
                "training_stage": self.stage}
        self.accel.save(ckp, save_pth)
        self.accel.print(f"Step {step}/{self.total_training_steps} | Training checkpoint saved at {save_pth}")

    def _load_checkpoint(self, load_pth, model_only=False):
        """load checkpoint after train objects are prepared by accel"""

        ckp = torch.load(load_pth, map_location="cpu")
        new_state_dict = OrderedDict()
        for key, value in ckp['model_state_dict'].items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        self.accel.unwrap_model(self.generator).load_state_dict(new_state_dict, strict=False)
        if model_only:
            self.accel.unwrap_model(self.optimizer).load_state_dict(ckp['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckp['scheduler_state_dict'])
            self.best_perf = ckp["best_perf"]
            self.start_step = ckp["step"] + 1
            self.stage = ckp["training_stage"]
            self.accel.print(f"Resume Training from Step {self.start_step}/{self.total_training_steps}")
            self.accel.print(f"Previous best Mel-Distance: {self.best_perf}")

    def load_train_objs(self, config, args):
        """Load Model, Optimizer, Scheduler, Metrics"""

        generator = make_model(config, vis=(self.accel.device=="cuda:0"), model="residual_csvq_codec")
        params = generator.parameters()
        optimizer = make_optimizer(params, optimizer_name="AdamW", lr=args.lr)
        scheduler = make_scheduler(optimizer, args.scheduler_type, total_steps=self.total_training_steps, warmup_steps=args.warmup_steps)
        metrics = make_metrics(self.accel.device)

        return generator, optimizer, scheduler, metrics

    def prepare_dataloader(self, args, config):
        """Load Dataloaders"""
        datasets = fetch_dataset("DNS_CHALLENGE", 
                                data_dir=config.data.data_dir, 
                                in_freq=config.data.in_freq,
                                win_len=config.data.win_len,
                                hop_len=config.data.hop_len,
                                sr=config.data.sr,
                                trans_on_cpu=args.trans_on_cpu)

        data_loaders = make_data_loader(datasets, 
                                        batch_size={"train": args.train_bs_per_device, "test": args.test_bs_per_device}, 
                                        shuffle={"train": True, "test": False}, 
                                        sampler=None, 
                                        num_workers=args.num_worker, verbose=True, seed=args.seed)

        train_steps_per_epoch, test_steps_per_epoch = len(data_loaders['train']) // args.num_device, len(data_loaders['test']) // args.num_device
        if self.args.eval_every == "epoch": self.args.eval_every = train_steps_per_epoch
        else: self.args.eval_every = int(self.args.eval_every)

        self.accel.print(f"batch_size_per_device: train {args.train_bs_per_device} test {args.test_bs_per_device}")
        self.accel.print(f"training_steps_per_dataset: {train_steps_per_epoch}, testing_steps_per_epoch: {test_steps_per_epoch}")
        self.accel.print(f"total_training_steps: {self.total_training_steps}")

        return data_loaders

    def run_naive(self):
        # Prepare steps for progressive training
        self.progressive_stages, self.progressive_cutoffs, self.total_training_steps = compute_full_cutoffs(
            self.config.training.warmup_steps, self.config.training.progressive_steps, self.config.training.stabilize_steps, self.config.training.dropout_steps)
        idx = self.progressive_stages.index(self.stage)
        self.current_cutoff = self.progressive_cutoffs[idx]
        self.next_stage = self.progressive_stages[idx+1]
        self.next_cutoff = self.progressive_cutoffs[idx+1]

        self.start_step, self.best_perf = 0, np.inf
        self.progress_bar = tqdm(initial=self.start_step, total=self.total_training_steps, position=0, leave=True)
        alphas = []
        while True:
            flag=False
            for _ in range(5000):
                
                self.check_progressive_stage(self.progress_bar.n)
                if self.stage == "dropout_step":
                    streams = quantization_dropout(self.args.q_dropout_rate, self.config.model.max_streams)
                    self.progress_bar.set_description(f"Training Model [{self.stage}][dropout sampling kbps]")
                else:
                    streams = int(self.stage.split("_")[-1]) + 1
                    self.progress_bar.set_description(f"Training Model [{self.stage}][{round(1.5*streams,1)}kbps]")
                alpha = self.compute_fade_in_alpha(self.progress_bar.n)
                alphas.append(alpha)
                self.progress_bar.update(1)
                if self.progress_bar.n == self.total_training_steps:
                    flag=True
                    break
            if flag: break

        plt.figure(figsize=(20,5))
        plt.plot(alphas)
        plt.savefig("../alpha_test.jpg")

def compute_full_cutoffs(warmup_steps: int=15000,
                         progressive_steps: list=[25000,25000,25000,25000,25000],
                         stabilize_steps: list=[15000,15000,15000,15000,15000],
                         dropout_steps: int=35000):
    
    maintain_step = warmup_steps
    progressive_cutoffs = {"stabilize_step_0": 0, "progressive_step_1": maintain_step}
    for i, prog_step in enumerate(progressive_steps):
        maintain_step += prog_step
        progressive_cutoffs[f"stabilize_step_{i+1}"] = maintain_step
        maintain_step += stabilize_steps[i]
        if i < len(progressive_steps)-1:
            progressive_cutoffs[f"progressive_step_{i+2}"] = maintain_step
        else:
            progressive_cutoffs["dropout_step"] = maintain_step

    total_training_steps = maintain_step + dropout_steps
    return list(progressive_cutoffs.keys()), list(progressive_cutoffs.values()), total_training_steps


def main(args, config):

    accel = Accelerator()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if accel.is_main_process:
        if args.wb_project_name is not None:
            wandb.login(key="df8e2c82f85049008e91367edf75e544588d2aa9")
            wandb.init(project=args.wb_project_name, name=args.wb_exp_name)
        else:   
            accel.print("Deactivated WandB Logging for Debugging")

    trainer = ProgressiveTrainer(accel, config, args)
    trainer.train()

    if accel.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    print(compute_full_cutoffs())