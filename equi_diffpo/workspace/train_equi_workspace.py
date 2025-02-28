if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import pickle
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from equi_diffpo.workspace.base_workspace import BaseWorkspace
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.dataset.base_dataset import BaseImageDataset
from equi_diffpo.env_runner.base_image_runner import BaseImageRunner
from equi_diffpo.common.checkpoint_util import TopKCheckpointManager
from equi_diffpo.common.json_logger import JsonLogger
from equi_diffpo.common.pytorch_util import dict_apply, optimizer_to
from equi_diffpo.model.diffusion.ema_model import EMAModel
from equi_diffpo.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainEquiWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: BaseImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: BaseImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        total_params = 0
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                total_params += param.numel()
        assert total_params == sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # configure training state
        self.global_step = 0
        #self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                #self.epoch += 1
                self.global_step += 1

        # configure dataset
        from equi_diffpo.dataset.real_dataset import load_data
        # dataset: BaseImageDataset
        # dataset = hydra.utils.instantiate(cfg.task.dataset)
        # assert isinstance(dataset, BaseImageDataset)
        # train_dataloader = DataLoader(dataset, **cfg.dataloader)
        # normalizer = dataset.get_normalizer()

        # # configure validation dataset
        # val_dataset = dataset.get_validation_dataset()
        # val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        train_dataloader, val_dataloader, stats= load_data(
            dataset_dir_l=cfg.dataset_path,
            robot_infor=cfg.robot_infor,  
            batch_size_train=cfg.dataloader.batch_size,
            batch_size_val=cfg.val_dataloader.batch_size,
            chunk_size=16,  
            use_depth_image=None,  
            sample_weights=None,  
            rank=None,  
            use_data_aug=None,  
            act_norm_class="norm1",  
            use_raw_lang=False,  
            exp_type=cfg.exp_type,  
            tg_mode="mode1"
        )
        #print(f"num_batches_per_epoch is:{num_batches_per_epoch}")
        
        # save dataset stats
        stats_path = os.path.join(self.output_dir,f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

        # self.model.set_normalizer(normalizer)
        # if cfg.training.use_ema:
        #     self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps= (cfg.training.train_steps) // cfg.training.gradient_accumulate_every,
            # num_training_steps=(
            #     len(train_dataloader) * cfg.training.num_epochs) \
            #         // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        
        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            step_log = dict()
            while self.global_step <= cfg.training.train_steps:
                #step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                # with tqdm.tqdm(train_dataloader, desc=f"Training step {self.global_step}", 
                #         leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                policy = self.model
                if (self.global_step != 0) and (self.global_step % cfg.training.val_every)==0:
                    # train_loss = np.mean(train_losses)
                    # step_log['train_loss'] = train_loss
                    print("here 1")
                        # ========= eval for this epoch ==========
                    policy = self.model
                    if cfg.training.use_ema:
                        policy = self.ema_model
                    policy.eval()

                    # run rollout
                    # if (self.global_step % cfg.training.rollout_every) == 0:
                    #     runner_log = env_runner.run(policy)
                    #     # log all
                    #     step_log.update(runner_log)

                    # run validation
                    
        
                    with torch.no_grad():
                        val_losses = list()
                        # with tqdm.tqdm(val_dataloader, desc=f"Validation step {self.global_step}", 
                        #         leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(val_dataloader):
                            
                            print(f"val step:{batch_idx}")
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            # if (cfg.training.max_val_steps is not None) \
                            #     and batch_idx >= (cfg.training.max_val_steps-1):
                            #     break
                            if batch_idx >= cfg.training.max_val_steps:
                                break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            print(f"val loss:{val_loss}")

                    # run diffusion sampling on a training batch
                    if (self.global_step % cfg.training.sample_every) == 0:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = batch['obs']
                            gt_action = batch['action']
                            
                            result = policy.predict_action(obs_dict)
                            pred_action = result['action_pred']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            step_log['train_action_mse_error'] = mse.item()
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            del mse
                    
                    # checkpoint
                    if (self.global_step % cfg.training.checkpoint_every) == 0:
                        # checkpointing
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

                        # sanitize metric names
                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value
                        
                        # We can't copy the last checkpoint here
                        # since save_checkpoint uses threads.
                        # therefore at this point the file might have been empty!
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)


                for batch_idx, batch in enumerate(train_dataloader):
                    if self.global_step > cfg.training.train_steps:
                        break
                    print(f"training step:{self.global_step}")
                    policy.train()
                    # # 打印每个batch的键
                    #print(batch.keys())
                    # # 打印obs的内容和类型
                    # print(f"obs content: {batch['obs']}, type: {type(batch['obs'])}")
                    # # 如果需要查看obs的键和大小
                    #print(f"obs keys: {batch['obs'].keys()}, obs size: {len(batch['obs'])}")
                    #print(f"1 action size: {batch['action'].size()}")
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    raw_loss = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    #tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    print(f"train_loss: {raw_loss_cpu}")
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        #'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }

                    
                    json_logger.log(step_log)
                    if batch_idx >= cfg.training.val_every:
                        train_loss = np.mean(train_losses)
                        step_log['train_loss'] = train_loss
                        break
                    self.global_step += 1
                    


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainEquiWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
