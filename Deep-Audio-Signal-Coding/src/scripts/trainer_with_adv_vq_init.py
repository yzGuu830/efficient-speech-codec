import os, torch, wandb, random, torchaudio, json
from data import make_data_loader, fetch_dataset
from scripts.utils import quantization_dropout, make_optimizer, make_scheduler, make_model, make_metrics, \
    Entropy_Counter, reset_codebooks

from models.discriminator import Discriminator
from models.losses.gan import GANLoss

import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from accelerate import Accelerator

class Trainer:
    def __init__(self, accel, config, args, ) -> None:
        
        self.args = args
        self.config = config
        self.accel = accel
        
        self.evaluation = None
        self.best_perf = None
        self.progress_bar = None
        self.entropy_counter = Entropy_Counter(codebook_size=config.model.codebook_size, 
                                               num_streams=config.model.max_streams, 
                                               num_groups=config.model.num_vqs, 
                                               device=accel.device)

        if not os.path.exists(f"{args.save_dir}/{args.wb_exp_name}"):
            os.makedirs(f"{args.save_dir}/{args.wb_exp_name}")
        self.accel.print(f"Saving outputs into {args.save_dir}/{args.wb_exp_name}")

    def _train_batch(self, input, epoch, i):
        if epoch <= self.args.pretrain_epochs: 
            streams, stage = self.config.model.max_streams, "pretrain at 0.00kbps"
        else:
            if epoch == self.args.pretrain_epochs+1 and i==0: 
                streams = self.config.model.max_streams # To initialize codebooks at all scales
            else:
                streams = quantization_dropout(self.args.q_dropout_rate, self.config.model.max_streams) \
                if self.is_scalable else self.config.model.max_streams
            stage = f"sampling at {(streams*1.5):.2f}kbps"
        self.progress_bar.set_description(f"Training Model [{stage}]")

        output = self.generator(**dict(x=input["audio"], 
                            x_feat=input["feat"] if "feat" in input else None, 
                            streams=streams, 
                            train=True,
                            freeze_codebook=(epoch<=self.args.pretrain_epochs)))

        if stage != "pretrain at 0.00kbps":
            # Update Discriminator
            self.discriminator.train()
            output["disc_loss"] = self.gan_loss.discriminator_loss(
                output["recon_audio"], output["raw_audio"]
            )
            self.optimizer_d.zero_grad()
            self.accel.backward(output["disc_loss"].mean())
            self.accel.clip_grad_norm_(self.discriminator.parameters(), 10.0)
            self.optimizer_d.step()

            # Update Generator
            output["gen_loss"], output["feat_loss"] = self.gan_loss.generator_loss(
                            output["recon_audio"], output["raw_audio"]  )
        else:
            output["disc_loss"] = torch.zeros(input["audio"].size(0), device=self.accel.device)
            output["gen_loss"], output["feat_loss"] = torch.zeros(input["audio"].size(0), device=self.accel.device), torch.zeros(input["audio"].size(0), device=self.accel.device)
        
        output["loss"] = self.config.loss.recon_factor * output["recon_loss"] + \
                            self.config.loss.commitment_factor * output["commitment_loss"] + \
                                self.config.loss.codebook_factor * output["codebook_loss"] + \
                                    self.config.loss.mel_factor * output["mel_loss"] + \
                                        self.config.loss.gen_factor * output["gen_loss"] + \
                                            self.config.loss.feat_factor * output["feat_loss"] + \
                                                self.config.loss.reg_factor * output["kl_loss"] 
        self.optimizer_g.zero_grad()
        self.accel.backward(output["loss"].mean())
        self.accel.clip_grad_norm_(self.generator.parameters(), 1e3)
        self.optimizer_g.step()
        self.scheduler.step()

        if self.evaluation is None:
            self.evaluation = {k: [] for k in output.keys() if k in ["loss", "recon_loss", 
                                                                     "commitment_loss", "codebook_loss", "mel_loss", "kl_loss",
                                                                     "disc_loss", "gen_loss", "feat_loss"]}
        for key, _ in self.evaluation.items():
            self.evaluation[key].append(output[key].mean().item())

        self.progress_bar.update(1)
        if self.progress_bar.n in self.args.save_steps:
            os.makedirs(f"{self.args.save_dir}/{self.args.wb_exp_name}/train_log/", exist_ok=True)
            torchaudio.save(f"{self.args.save_dir}/{self.args.wb_exp_name}/train_log/raw_audio_step_{self.progress_bar.n}.wav",
                            input["audio"][0:1].cpu(), 16000)
            torchaudio.save(f"{self.args.save_dir}/{self.args.wb_exp_name}/train_log/recon_audio_step_{self.progress_bar.n}.wav",
                            output["recon_audio"][0:1].cpu(), 16000)

    def _test_batch(self, input, epoch):
        self.generator.eval()
        output = self.generator(**dict(x=input["audio"], 
                            x_feat=input["feat"] if "feat" in input else None, 
                            streams=self.config.model.max_streams, 
                            train=False,
                            freeze_codebook=(epoch<=self.args.pretrain_epochs)))
        local_obj_metric = {} # metric stats on each device
        for k, m in self.obj_metric.items():
            if k in ['Test_PESQ', 'Test_MelDist', 'Test_STFTDist', 'Test_SNR']:
                local_obj_metric[k] = m(input["audio"], output["recon_audio"]) # Tensor
            elif k in ["Test_PSNR"]:
                local_obj_metric[k] = m(output["raw_feat"], output["recon_feat"]) # Tensor

        local_obj_metric_gathered = {} # metric stats gathered from all device
        for k, v in local_obj_metric.items():
            local_obj_metric_gathered[k] = self.accel.gather(v.to(self.accel.device)).tolist()

        gathered = self.accel.gather(torch.stack(output["multi_codes"],dim=-1)) # bs*num_device G T streams (streams=1 when pretrain)
        if self.accel.is_main_process and epoch > self.args.pretrain_epochs:
            self.entropy_counter.update(multi_codes=gathered)

        for k, v_g in local_obj_metric_gathered.items(): # cummulate for all batches
            self.cumm_metric[k].extend(v_g)

    def _log_train_batch(self):
        for key, val in self.evaluation.items():
            self.evaluation[key] = np.mean(val)
        if wandb.run is not None:
            wandb.log(self.evaluation)
        self.evaluation = None

    def _log_test_batch(self, epoch):
        test_performance = {}
        for k, v in self.cumm_metric.items():
            test_performance[k] = np.mean(v)
        if epoch > self.args.pretrain_epochs:
            test_performance["bitrate_efficiency"] = self.entropy_counter.compute_bitrate_efficiency(return_total=True)
        else:
            test_performance["bitrate_efficiency"] = -1.0 # no quantizatio
        if wandb.run is not None:
            wandb.log(test_performance)
        return test_performance 

    def _train_epoch(self, epoch):
        # At start of finetune stages, one-time initialization of codebooks, set trigger to 0
        if epoch == self.args.pretrain_epochs+1: 
            for gvq in self.accel.unwrap_model(self.generator).quantizer:
                reset_codebooks(gvq) # set initialize flag to false (to re-initialize the codebooks)
                gvq.verbose_init = self.accel.is_main_process # verbose initialization at main device
                gvq.codebook_initialized.fill_(0)
            
            # renew optimizer and scheduler
            self.accel.print(f"Renew Optimizer: AdamW w lr={self.args.lr*0.3} and Scheduler: {self.args.scheduler_type}_decay")
            optimizer = make_optimizer(self.generator.parameters(), optimizer_name="AdamW", lr=self.args.lr*0.3)
            self.scheduler = make_scheduler(
                                optimizer, 
                                self.args.scheduler_type, 
                                total_steps=self.args.max_train_steps-(self.args.pretrain_epochs*len(self.train_data)), 
                                warmup_steps=self.args.warmup_steps)
            self.optimizer = self.accel.prepare(optimizer)
            
        elif (self.args.pretrain_epochs!=0) and epoch == 1:
            # During pretrain stages set trigger to 1 to prevent initialize
            for gvq in self.accel.unwrap_model(self.generator).quantizer:
                gvq.codebook_initialized.fill_(1)

        for i, input in enumerate(self.train_data):
            self._train_batch(input, epoch, i)
            if (i+1) % self.args.info_steps == 0 and self.accel.is_main_process:
                self._log_train_batch()

    def _test_epoch(self, epoch):
        """Distributed Evaluate"""
        self.cumm_metric = {k:[] for k in self.obj_metric.keys()}
        if self.accel.is_main_process:
            self.entropy_counter.reset()
            for _, input in tqdm(enumerate(self.test_data), total=len(self.test_data), desc=f"Eval Epoch {epoch}"): 
                self._test_batch(input, epoch)
        else:
            for _, input in enumerate(self.test_data): 
                self._test_batch(input, epoch)

        if self.accel.is_main_process:
            test_performance = self._log_test_batch(epoch)
            self.accel.print(" | ".join(f"{k}: {v:.4f}" for k, v in test_performance.items()))
            
            if epoch > self.args.pretrain_epochs: # Finetuning Stage
                if test_performance["Test_MelDist"] < self.best_perf:
                    self.best_perf = test_performance["Test_MelDist"]
                    self.accel.print(f"Found Best Model at epoch {epoch}")
                    os.makedirs(f"{self.args.save_dir}/{self.args.wb_exp_name}", exist_ok=True)
                    self._save_checkpoint(epoch, save_pth=f"{self.args.save_dir}/{self.args.wb_exp_name}/best.pt")
                stage = "finetune"
                # save vq utility stats
                multi_efficiency = self.entropy_counter.compute_bitrate_efficiency(return_total=False)
                root_path = f"{self.args.save_dir}/{self.args.wb_exp_name}/train_log"
                if not os.path.exists(root_path): os.makedirs(root_path)
                if not os.path.exists(f"{root_path}/codebook_util_stats.json"):
                    json.dump({f"epoch {epoch}": multi_efficiency}, open(f"{root_path}/codebook_util_stats.json", "w"), indent=4)
                else:
                    stats = json.load(open(f"{root_path}/codebook_util_stats.json", "r"))
                    stats[f"epoch {epoch}"] = multi_efficiency
                    json.dump(stats, open(f"{root_path}/codebook_util_stats.json", "w"), indent=4)
            else:
                stage = "pretrain"
            os.makedirs(f"{self.args.save_dir}/{self.args.wb_exp_name}/{stage}", exist_ok=True)
            self._save_checkpoint(epoch, save_pth=f"{self.args.save_dir}/{self.args.wb_exp_name}/{stage}/checkpoint.pt")

        self.accel.wait_for_everyone()

    def train(self, max_epochs: int):
        """main training method"""
        # Prepare Dataloaders
        dls = self.prepare_dataloader(self.args, self.config)
        self.train_data = self.accel.prepare(dls["train"])
        self.test_data = self.accel.prepare(dls["test"])        
        # Prepare Model Optimizer Scheduler Metrics
        self.generator, self.optimizer_g, self.discriminator, self.optimizer_d, self.scheduler, self.obj_metric = self.load_train_objs(self.config, self.args)

        self.is_scalable = self.generator.scalable
        self.accel.print(f"Running Experiment: {self.args.wb_exp_name}")
        self.accel.print(f"Learning_rate: {self.args.lr} | Batch_size: {self.args.train_bs_per_device * self.args.num_device} | Scalable: {self.is_scalable}")
        self.accel.print(f"Quantizer_dropout_rate: {self.args.q_dropout_rate} | Num_worker: {self.args.num_worker} | Pretrain_epochs: {self.args.pretrain_epochs}")
        init_map = {None: "No Initialization", True: "Kmeans++ from z_e", False: "RandomSelect from z_e"}
        self.accel.print("Codebook initialization approach: {}".format(init_map[self.config.model.kmeans_init]))
        
        # Initialize from a pretrained autoencoder
        if self.args.init_ckpt is not None:
            if os.path.exists(self.args.init_ckpt):
                self.accel.print(f"Initialize EncoderDecoder from {self.args.init_ckpt}")
                self._load_checkpoint(self.args.init_ckpt, generator_only=True)
        else:
            self.accel.print("Start Training From Scratch")
            self.start_epoch, self.best_perf = 1, np.inf 
        
        self.generator, self.optimizer_g, self.discriminator, self.optimizer_d = self.accel.prepare(
            self.generator, self.optimizer_g, self.discriminator, self.optimizer_d )
        self.gan_loss = GANLoss(self.discriminator)
        self.progress_bar = tqdm(initial=(self.start_epoch-1)*len(self.train_data), 
                                 total=max_epochs*len(self.train_data), position=0, leave=True)
        for epoch in range(self.start_epoch, max_epochs+1):
            self.accel.print(f"---Epoch {epoch} Training---")
            self._train_epoch(epoch)
            self._test_epoch(epoch)

    def _save_checkpoint(self, epoch, save_pth):
        """Save accel.prepare(object) checkpoints"""
        # self.accel.wait_for_everyone()
        ckp = {'epoch': epoch, 
            'model_state_dict': self.accel.unwrap_model(self.generator).state_dict(),
            'disc_state_dict': self.accel.unwrap_model(self.discriminator).state_dict(),
            'optimizer_g_state_dict': self.accel.unwrap_model(self.optimizer_g).state_dict(), 
            'optimizer_d_state_dict': self.accel.unwrap_model(self.optimizer_d).state_dict(), 
            'scheduler_state_dict': self.scheduler.state_dict(),
            "best_perf": self.best_perf}
        
        self.accel.save(ckp, save_pth)
        self.accel.print(f"Epoch {epoch} | Training checkpoint saved at {save_pth}")

    def _load_checkpoint(self, load_pth, generator_only=True):
        """load checkpoint after train objects are prepared by accel"""
        ckp = torch.load(load_pth, map_location="cpu")
        new_state_dict = OrderedDict()
        for key, value in ckp['model_state_dict'].items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        self.accel.unwrap_model(self.generator).load_state_dict(new_state_dict)

        if not generator_only:
            new_state_dict = OrderedDict()
            for key, value in ckp['disc_state_dict'].items():
                if key.startswith('module.'):
                    new_state_dict[key[7:]] = value
                else:
                    new_state_dict[key] = value
            self.accel.unwrap_model(self.discriminator).load_state_dict(new_state_dict)
            self.accel.unwrap_model(self.optimizer_d).load_state_dict(ckp['optimizer_d_state_dict'])

        self.accel.unwrap_model(self.optimizer_g).load_state_dict(ckp['optimizer_g_state_dict'])
        self.scheduler.load_state_dict(ckp['scheduler_state_dict'])
        self.best_perf = ckp["best_perf"]
        self.start_epoch = ckp["epoch"] + 1

        self.accel.print(f"Resume Training from Epoch {self.start_epoch}")
        self.accel.print(f"Previous best MelDist: {self.best_perf}")

    def load_train_objs(self, config, args):
        """Load Model, Optimizer, Scheduler, Metrics"""

        generator = make_model(config, vis=(self.accel.device=="cuda:0"))
        params = generator.parameters()
        optimizer_g = make_optimizer(params, optimizer_name="AdamW", lr=args.lr)

        discriminator = Discriminator(rates=config.discriminator.rates, 
                                      periods=config.discriminator.periods,
                                      fft_sizes=config.discriminator.fft_sizes,
                                      sample_rate=config.discriminator.sample_rate,
                                      bands=config.discriminator.bands)
        optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr)

        scheduler = make_scheduler(optimizer_g, args.scheduler_type, total_steps=args.max_train_steps, warmup_steps=args.warmup_steps)
        metrics = make_metrics(self.accel.device)

        return generator, optimizer_g, discriminator, optimizer_d, scheduler, metrics

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