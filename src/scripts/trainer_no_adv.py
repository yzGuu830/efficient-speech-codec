import os, torch, wandb, random, torchaudio
from data import make_data_loader, fetch_dataset
from utils import quantization_dropout, calculate_stage_cutoffs, maintain_stage, \
    make_optimizer, make_scheduler, make_model, make_metrics, switch_stage

import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from accelerate import Accelerator, DistributedDataParallelKwargs

class Trainer:
    def __init__(self, accel, config, args, ) -> None:
        
        self.args = args
        self.config = config
        self.accel = accel
        
        self.evaluation = None
        self.best_perf = None
        self.progress_bar = None
        self.stage = None

        if not os.path.exists(f"{args.save_dir}/{args.wb_exp_name}"):
            os.makedirs(f"{args.save_dir}/{args.wb_exp_name}")
        self.accel.print(f"Saving outputs into {args.save_dir}/{args.wb_exp_name}")
            
    def _train_batch(self, input, i):
        self.stage = maintain_stage(self.progress_bar.n, self.adap_training_steps)
        self.progress_bar.set_description(f"Training Model | Stage: {self.stage}")
        
        streams = quantization_dropout(self.args.q_dropout_rate, self.config.model.max_streams) \
            if self.is_scalable else self.config.model.max_streams

        if self.stage == "warmup":  streams = 0 # During warmup, no quantizers are trained

        output = self.generator(**dict(x=input["audio"], 
                            x_feat=input["feat"] if "feat" in input else None, 
                            streams=streams, 
                            train=True))

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

        if self.evaluation is None:
            self.evaluation = {k: [] for k in output.keys() if k in ["loss", "recon_loss", 
                                                                     "commitment_loss", "codebook_loss", "mel_loss", "kl_loss",
                                                                     ]}
        for key, _ in self.evaluation.items():
            self.evaluation[key].append(output[key].mean().item())

        self.progress_bar.update(1)
        if self.progress_bar.n in self.args.save_steps:
            os.makedirs(f"{self.args.save_dir}/{self.args.wb_exp_name}/train_log/", exist_ok=True)
            torchaudio.save(f"{self.args.save_dir}/{self.args.wb_exp_name}/train_log/raw_audio_step_{self.progress_bar.n}.wav",
                            input["audio"][0:1].cpu(), 16000)
            torchaudio.save(f"{self.args.save_dir}/{self.args.wb_exp_name}/train_log/recon_audio_step_{self.progress_bar.n}.wav",
                            output["recon_audio"][0:1].cpu(), 16000)

        if self.progress_bar.n == self.adap_training_steps[0]:
            self.accel.print("Transition from Warmup to Freeze Training")
            generator, optimizer = switch_stage(self.accel.unwrap_model(self.generator), "warmup->freeze", "AdamW", self.args.lr)
            self.generator, self.optimizer = self.accel.prepare(generator, optimizer)
        elif self.progress_bar.n == self.adap_training_steps[1]:
            self.accel.print("Transition from Freeze to Refine Training")
            generator, optimizer = switch_stage(self.accel.unwrap_model(self.generator), "freeze->refine", "AdamW", self.args.lr)
            self.generator, self.optimizer = self.accel.prepare(generator, optimizer)

    def _test_batch(self, input):
        self.generator.eval()
        output = self.generator(**dict(x=input["audio"], 
                            x_feat=input["feat"] if "feat" in input else None, 
                            streams=self.config.model.max_streams if not self.warmup_training else 0, 
                            train=False))
        local_obj_metric = {} # metric stats on each device
        for k, m in self.obj_metric.items():
            if k in ['Test_PESQ', 'Test_MelDist', 'Test_STFTDist', 'Test_SNR']:
                local_obj_metric[k] = m(input["audio"], output["recon_audio"])
            elif k in ["Test_PSNR"]:
                local_obj_metric[k] = m(output["raw_feat"], output["recon_feat"])

        local_obj_metric_gathered = {} # metric stats gathered from all device
        for k, v in local_obj_metric.items():
            local_obj_metric_gathered[k] = self.accel.gather(
                torch.tensor(v, device=self.accel.device)
            ).tolist()

        for k, v_g in local_obj_metric_gathered.items(): # cummulate for all batches
            self.cumm_metric[k].extend(v_g)

    def _log_train_batch(self):
        for key, val in self.evaluation.items():
            self.evaluation[key] = np.mean(val)
        if wandb.run is not None:
            wandb.log(self.evaluation)
        self.evaluation = None

    def _log_test_batch(self):
        test_performance = {}
        for k, v in self.cumm_metric.items():
            test_performance[k] = np.mean(v)
        if wandb.run is not None:
            wandb.log(test_performance)
        return test_performance 

    def _train_epoch(self, epoch):
        
        for i, input in enumerate(self.train_data):
            
            self._train_batch(input, i)

            if (i+1) % self.args.info_steps == 0 and self.accel.is_main_process:
                self._log_train_batch()

    def _test_epoch(self, epoch):
        """Distributed Evaluate"""
        self.cumm_metric = {k:[] for k in self.obj_metric.keys()}
        if self.accel.is_main_process:
            for _, input in tqdm(enumerate(self.test_data), total=len(self.test_data), desc=f"Eval Epoch {epoch}"): 
                self._test_batch(input)
        else:
            for _, input in enumerate(self.test_data): 
                self._test_batch(input)

        if self.accel.is_main_process:
            test_performance = self._log_test_batch()
            self.accel.print(" | ".join(f"{k}: {v:.4f}" for k, v in test_performance.items()))
            if test_performance["Test_PESQ"] > self.best_perf:
                self.best_perf = test_performance["Test_PESQ"]
                self.accel.print(f"Found Best Model at epoch {epoch}")
                self._save_checkpoint(epoch, save_pth=f"{self.args.save_dir}/{self.args.wb_exp_name}/{self.stage}/best.pt")
            self._save_checkpoint(epoch, save_pth=f"{self.args.save_dir}/{self.args.wb_exp_name}/{self.stage}/checkpoint.pt")

    def _run_epoch(self, epoch):
        if self.accel.is_main_process:
            self.accel.print(f"---Epoch {epoch} Training---")
        self._train_epoch(epoch)
        self._test_epoch(epoch)

    def train(self, max_epochs: int):
        """main training method"""
        # Prepare Dataloaders
        dls = self.prepare_dataloader(self.args, self.config)
        self.train_data = self.accel.prepare(dls["train"])
        self.test_data = self.accel.prepare(dls["test"])
        # Prepare Model Optimizer Scheduler Metrics 
        generator, optimizer, self.scheduler, self.obj_metric = self.load_train_objs(self.config, self.args)
        self.generator, self.optimizer = self.accel.prepare(generator, optimizer)

        # Prepare Adap Training
        self.adap_training_steps = calculate_stage_cutoffs(self.args.max_train_steps, self.args.training_fractions)
        self.is_scalable = generator.scalable
        self.accel.print(f"Running Experiment: {self.args.wb_exp_name}")
        self.accel.print(f"learning_rate: {self.args.lr} | batch_size: {self.args.train_bs_per_device * self.args.num_device} | scalable: {self.is_scalable}")
        self.accel.print(f"quantize_dropout_rate: {self.args.q_dropout_rate} | augment: {self.args.augment} | num_worker: {self.args.num_worker}")
        self.accel.print("Number of Training (K)steps for 'warmup' | 'freeze' | 'refine': {}".format(
            " | ".join([str(round(s*self.args.max_train_steps/1000, 1)) for s in self.args.training_fractions]) ))

        # Resume Training from some Stage
        if self.args.resume_from in ["warmup", "freeze", "refine"]:
            resumed_checkpoint_pth = f"{self.args.save_dir}/{self.args.wb_exp_name}/{self.args.resume_from}/checkpoint.pt"
            if os.path.exists(resumed_checkpoint_pth):
                print(f"Resume Training from {self.args.resume_from}")
                self._load_checkpoint(resumed_checkpoint_pth)
            else:
                raise ValueError(f"Did not find any pre-trained checkpoints from {resumed_checkpoint_pth}")
        else:
            self.accel.print("Start Adap Training From Scratch")
            self.start_epoch, self.best_perf = 1, -np.inf            
        
        self.progress_bar = tqdm(initial=(self.start_epoch-1)*len(self.train_data), 
                                 total=max_epochs*len(self.train_data), position=0, leave=True)
        for epoch in range(self.start_epoch, max_epochs+1):
            self._run_epoch(epoch) 

    def _save_checkpoint(self, epoch, save_pth):
        """Save accel.prepare(object) checkpoints"""
        # self.accel.wait_for_everyone()
        ckp = {
            'epoch': epoch, 
            'model_state_dict': self.accel.unwrap_model(self.generator).state_dict(),
            'optimizer_state_dict': self.accel.unwrap_model(self.optimizer).state_dict(), 
            'scheduler_state_dict': self.scheduler.state_dict(),
            "best_perf": self.best_perf}
        
        self.accel.save(ckp, save_pth)
        self.accel.print(f"Epoch {epoch} | Training checkpoint saved at {save_pth}")

    def _load_checkpoint(self, load_pth):
        """load checkpoint after train objects are prepared by accel"""

        ckp = torch.load(load_pth, map_location="cpu")
        new_state_dict = OrderedDict()
        for key, value in ckp['model_state_dict'].items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        self.accel.unwrap_model(self.generator).load_state_dict(new_state_dict)

        self.accel.unwrap_model(self.optimizer).load_state_dict(ckp['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckp['scheduler_state_dict'])
        self.best_perf = ckp["best_perf"]
        self.start_epoch = ckp["epoch"] + 1

        self.accel.print(f"Resume Training from Epoch {self.start_epoch}")
        self.accel.print(f"Previous best PESQ: {self.best_perf}")

    def load_train_objs(self, config, args):
        """Load Model, Optimizer, Scheduler, Metrics"""

        generator = make_model(config, vis=(self.accel.device=="cuda:0"))
        params = generator.parameters()
        optimizer = make_optimizer(params, optimizer_name="AdamW", lr=args.lr)
        scheduler = make_scheduler(optimizer, args.scheduler_type, total_steps=args.max_train_steps, warmup_steps=args.warmup_steps)
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
        max_train_steps = train_steps_per_epoch*args.num_epochs
        self.args.train_steps_per_epoch = train_steps_per_epoch
        self.args.max_train_steps = max_train_steps
        self.accel.print(f"batch_size_per_device: train {args.train_bs_per_device} test {args.test_bs_per_device}")
        self.accel.print(f"training_steps_per_epoch: {train_steps_per_epoch}, testing_steps_per_epoch: {test_steps_per_epoch}")
        self.accel.print(f"total_training_steps: {max_train_steps}")

        return data_loaders


def main(args, config):

    accel = Accelerator()
    # kwarg = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accel = Accelerator(kwargs_handlers=[kwarg])
    # For Reproducibility
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

    trainer = Trainer(accel, config, args)
    trainer.train(args.num_epochs)

    if accel.is_main_process:
        wandb.finish()

