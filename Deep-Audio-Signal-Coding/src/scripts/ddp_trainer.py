
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os, torch, wandb, transformers, random

from models.codec import SwinAudioCodec
from data import make_data_loader, fetch_dataset
from utils import PESQ

import numpy as np
from tqdm import tqdm
from collections import OrderedDict


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
class Trainer:
    def __init__(self, gpu_id: int, config, args, ) -> None:
        
        self.gpu_id = gpu_id
        self.args = args
        self.config = config

        model, self.optimizer, self.scheduler = self.load_train_objs(config, args)
        self.is_scalable = model.scalable

        model = model.to(gpu_id)
        dls = self.prepare_dataloader(args, config)
        self.train_data = dls["train"]
        self.test_data = dls["test"]
        
        self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
        
        self.evaluation = None
        self.best_perf = None
        self.progress_bar = None

    def _train_batch(self, input):
        if self.is_scalable:
            streams = np.random.randint(1, self.config.model.max_streams+1)

        self.optimizer.zero_grad()
        output = self.model(x=input["audio"], 
                            x_feat=input["feat"], 
                            streams=streams, 
                            train=True)
        output["loss"] = self.config.loss.recon_factor * output["recon_loss"] + \
                            self.config.loss.commitment_factor * output["commitment_loss"] + \
                                self.config.loss.codebook_factor * output["codebook_loss"] + \
                                    self.config.loss.mel_factor * output["mel_loss"]
        
        loss = output["loss"].mean()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        self.optimizer.step()
        self.scheduler.step()

        if self.gpu_id == 0:
            if self.evaluation is None:
                self.evaluation = {k: [] for k in output.keys() if k in ["loss", "recon_loss", "commitment_loss", "codebook_loss", "mel_loss"]}
            for key, _ in self.evaluation.items():
                self.evaluation[key].append(output[key].mean().item())

            self.progress_bar.update(1)

    def _test_batch(self, input):
        output = self.model(x=input["audio"], 
                            x_feat=input["feat"], 
                            streams=self.config.model.max_streams, 
                            train=False)
        self.local_obj_metric.extend(
            [PESQ(input['audio'][j].cpu().numpy(), 
                  output['recon_audio'][j].cpu().numpy()) for j in range(input['audio'].size(0))]
        )

    def _log_train_batch(self):
        for key, val in self.evaluation.items():
            self.evaluation[key] = np.mean(val)
        if wandb.run is not None:
            wandb.log(self.evaluation)
        self.evaluation = None

    def _log_test_batch(self):
        all_metrics = torch.cat(self.gathered_metrics)
        test_performance = all_metrics.mean().item()
        if wandb.run is not None:
            wandb.log({"Test_PESQ": test_performance})
        return test_performance 

    def _train_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        for i, input in enumerate(self.train_data):
            for key in input.keys():
                if input[key] is not None:
                    input[key] = input[key].to(self.gpu_id)
            self._train_batch(input)

            if (i+1) % self.args.info_steps == 0 and self.gpu_id == 0:
                self._log_train_batch()


    def _test_epoch(self, epoch):
        self.local_obj_metric = []
        for _, input in enumerate(self.test_data): 
            for key in input.keys():
                if input[key] is not None:
                    input[key] = input[key].to(self.gpu_id)
            self._test_batch(input)

        # Gather test performance on all devices
        self.local_obj_metric = torch.tensor(self.local_obj_metric, device=f"cuda:{self.gpu_id}")
        self.gathered_metrics = [torch.zeros_like(self.local_obj_metric) for _ in range(dist.get_world_size())]
        dist.all_gather(self.gathered_metrics, self.local_obj_metric)

        if self.gpu_id == 0:
            test_performance = self._log_test_batch()
            print(f"Test PESQ: {test_performance:.4f}")
            if test_performance > self.best_perf:
                self.best_perf = test_performance
                print(f"Found Best Model at epoch {epoch}")
                self._save_checkpoint(epoch, save_pth=f"{self.args.save_dir}/{self.args.wb_exp_name}/best.pt")
            self._save_checkpoint(epoch, save_pth=f"{self.args.save_dir}/{self.args.wb_exp_name}/checkpoint.pt")
        
        dist.barrier()

    def _run_epoch(self, epoch):
        b_sz = next(iter(self.train_data))["audio"].size(0)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Train Batchsize: {b_sz} | Train Steps: {len(self.train_data)}")
        self._train_epoch(epoch)
        self._test_epoch(epoch)

    def train(self, max_epochs: int):

        resumed_checkpoint_pth = f"{self.args.save_dir}/{self.args.wb_exp_name}/checkpoint.pt"
        if os.path.exists(resumed_checkpoint_pth):
            self._load_checkpoint(resumed_checkpoint_pth)
        else:
            self.start_epoch, self.best_perf = 1, -np.inf
        
        self.progress_bar = tqdm(initial=(self.start_epoch-1)*len(self.train_data), 
                                 total=max_epochs*len(self.train_data), position=0, leave=True,
                                 desc="Training Model") if self.gpu_id == 0 else None
        for epoch in range(self.start_epoch, max_epochs+1):
            self._run_epoch(epoch)

    def _save_checkpoint(self, epoch, save_pth):
        ckp = {'epoch': epoch, 
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), 
            'scheduler_state_dict': self.scheduler.state_dict(),
            "best_perf": self.best_perf}
        
        torch.save(ckp, save_pth)
        print(f"Epoch {epoch} | Training checkpoint saved at {save_pth}")

    def _load_checkpoint(self, load_pth):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.gpu_id}
        ckp = torch.load(load_pth, map_location=map_location)
        new_state_dict = OrderedDict()
        for key, value in ckp['model_state_dict'].items():
            if not key.startswith('module.'):
                new_state_dict['module.' + key] = value
            else:
                new_state_dict[key] = value
        self.model.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(ckp['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckp['scheduler_state_dict'])
        self.best_perf = ckp["best_perf"]
        self.start_epoch = ckp["epoch"] + 1
        if self.gpu_id==0:
            print(f"Resume Training from Epoch {self.start_epoch}")
            print(f"Previous best PESQ: {self.best_perf}")

    def load_train_objs(self, config, args):
        if self.gpu_id == 0: 
            if not os.path.exists(f"{args.save_dir}/{args.wb_exp_name}"):
                os.makedirs(f"{args.save_dir}/{args.wb_exp_name}")
            print(f"Saving outputs into {args.save_dir}/{args.wb_exp_name}")

        model = SwinAudioCodec(config.model.in_dim, config.model.in_freq, config.model.h_dims,
                               config.model.swin_depth, config.model.swin_heads, config.model.window_size, 
                               config.model.mlp_ratio, config.model.max_streams, config.model.overlap, 
                               config.model.num_vqs, config.model.codebook_size, config.model.codebook_dims,
                               config.model.patch_size, config.model.use_ema, config.model.use_cosine_sim, 
                               config.model.is_causal, config.model.fuse_net, config.model.scalable,
                               config.model.mel_windows, config.model.mel_bins, config.model.win_len,
                               config.model.hop_len, config.model.sr, vis=(self.gpu_id==0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        if args.scheduler_type == "constant":
            scheduler = transformers.get_constant_schedule(optimizer)
        elif args.scheduler_type == "constant_warmup":
            scheduler = transformers.get_constant_schedule_with_warmup(optimizer,
                                                                    num_warmup_steps=args.warmup_steps) 
        elif args.scheduler_type == "cosine_warmup":
            transformers.get_cosine_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps=args.warmup_steps, 
                                                        num_training_steps=args.max_train_steps)
        return model, optimizer, scheduler

    def prepare_dataloader(self, args, config):

        datasets = fetch_dataset("DNS_CHALLENGE", 
                                data_dir=config.data.data_dir, 
                                in_freq=config.data.in_freq,
                                win_len=config.data.win_len,
                                hop_len=config.data.hop_len,
                                sr=config.data.sr)

        data_loaders = make_data_loader(datasets, 
                                        batch_size={"train": args.train_bs_per_device, "test": args.test_bs_per_device}, 
                                        shuffle={"train":False, "test":False}, 
                                        sampler={"train": DistributedSampler(datasets["train"], num_replicas=args.num_device, rank=self.gpu_id, shuffle=False, drop_last=False), 
                                                "test": DistributedSampler(datasets["test"], num_replicas=args.num_device, rank=self.gpu_id, shuffle=False, drop_last=False)}, 
                                        num_workers=args.num_worker, verbose=(self.gpu_id==0), seed=args.seed)

        train_steps_per_epoch, test_steps_per_epoch = len(data_loaders['train']), len(data_loaders['test'])
        max_train_steps = train_steps_per_epoch*args.num_epochs
        self.args.max_train_steps = max_train_steps
        if self.gpu_id == 0:
            print(f"batch_size_per_device: train {args.train_bs_per_device} test {args.test_bs_per_device}")
            print(f"training_steps_per_epoch: {train_steps_per_epoch}, testing_steps_per_epoch: {test_steps_per_epoch}")
            print(f"total_training_steps: {max_train_steps}")

        return data_loaders


def main(rank: int, 
         world_size: int, 
         args,
         config):
    
    # For Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ddp_setup(rank, world_size)
    trainer = Trainer(rank, config, args)

    if rank==0:
        if args.wb_project_name is not None:
            wandb.login(key="880cb5a13d061af184bd6f3833bbce3df6d099fc")
            wandb.init(project=args.wb_project_name, name=args.wb_exp_name)
        else:   print("Deactivated WandB Logging for Debugging")

    trainer.train(args.num_epochs)

    destroy_process_group()
    if rank==0:
        wandb.finish()



