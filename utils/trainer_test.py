

import torch
# import nni
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from torch.nn import MSELoss, SmoothL1Loss
import numpy as np
from pathlib import Path
import mlflow
from tqdm import tqdm
import shutil
import os
from collections import defaultdict
import ipdb
from .utils import is_parallel
from .std_logger import Logger
from .loss import TolerantMSELossv2
import time

class Trainer(object):

    def __init__(self, net, dataloaders, optimizer, scheduler, metrics, feature_fetcher,
                 device, global_rank, cfg, best_model_dir = None):

        self.net = net
        self.feature_fetcher = feature_fetcher
        self.device = device
        # self.criterion = criterion
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epoch = cfg.train.num_epoch
        self.global_rank = global_rank
        
        self.default_metric = cfg.task[cfg.task.type].default_metric

        self.cfg = cfg

        self.global_train_step = 0
        self.global_valid_eval_epoch = 0
        self.global_train_eval_epoch = 0
        # self.global_test_eval_epoch = 0

        # metrics
        self.metrics_func = metrics

        if cfg.train.loss =='mse':
            self.reg_loss_func = MSELoss()
        if cfg.train.loss =='huber':
            self.reg_loss_func = SmoothL1Loss(beta=cfg.train.loss_beta)
        if cfg.train.loss =='tmse':
            self.reg_loss_func = TolerantMSELossv2(margin=cfg.train.loss_beta)
        # save checkpoint
        self.best_metric = -1
        self.best_model_path = Path('.')
        self.best_model_dir = best_model_dir

        self.root_level_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def evaluate(self,split):
        self.net.eval()
        
        # loss_dict = defaultdict(list)
        # y_true_list = []
        # y_pred_list = []
        
        with torch.no_grad():
            # total_loss = []
            loss = 0
            y_pred, y_true = [],[]
            # latent = []
            # metrics = {}
            for step, data in tqdm(
                    enumerate(self.dataloaders[split]),
                    total=len(self.dataloaders[split]),
                    desc="evaluating {}| loss: {:.4f}".format(split,loss)):

                sequence, name2id, label = data
                
                # feature_fetcher here
                token_sequence = self.feature_fetcher.query_features(sequence['peptide'])
                
                batch = {k: torch.tensor(v).to(self.device) for k, v in token_sequence.items()}
                # ipdb.set_trace()
                noised_labels = torch.tensor(label[f'noised_{self.cfg.task.type}']).to(self.device)
                true_labels = torch.tensor(label[self.cfg.task.type]).to(self.device)
                
                output = self.net(batch)
                
                loss = self.reg_loss_func(output[0].squeeze(),noised_labels.squeeze())

                y_pred.append(output[0])
                y_true.append(true_labels)
                # latent.append(output.hidden_states[-1])
                # y_pred = output.logits
                # y_true = batch['labels']
                # total_loss.append(loss)
                # batch_metrics = self.metrics_func(y_pred, y_true,split)

                # for k, v in batch_metrics.items():
                #     if k not in metrics:
                #         metrics[k] = [v]
                #     else:
                #         metrics[k].append(v)
                # mask_pos.extend(label['pos'])
            y_pred = torch.cat(y_pred)
            y_true = torch.cat(y_true)
            
        metrics = self.metrics_func(y_pred, y_true,split)
        # ipdb.set_trace()
        
        res_dict = {}
        # for k,v in loss_dict.items():
        #     res_dict["{}_{}".format(split, k )] = np.mean(v)
        # res_dict['loss'] = np.mean(total_loss)
        for k, v in metrics.items():
            res_dict[f"{split}-{k}"] = np.mean(v)
            
        return res_dict

    def eval_epoch(self,split):

        metrics = self.evaluate(split)

        if split == "valid":
            self.global_valid_eval_epoch += 1
            step = self.global_valid_eval_epoch
        elif split == "train":
            self.global_train_eval_epoch += 1
            step = self.global_train_eval_epoch

        if self.global_rank == 0:

            for metric_name, metric_v in metrics.items():
                if isinstance(metric_v,  (float, np.float64, int, np.int_)):
                    metric_v = round(metric_v, 5)
                elif isinstance(metric_v,  str):
                    metric_v = "\n" + metric_v
                Logger.info("{} | step: {} | {}: {}".format(split,step, metric_name, metric_v))
            

            if not self.cfg.mode.nni and self.global_rank == 0 and self.cfg.logger.log:
                for metric_name, metric_v in metrics.items():
                    if isinstance(metric_v,  (float, np.float64, int, np.int_)):
                        mlflow.log_metric("{}_eval/{}".format(split,metric_name),
                                        metric_v,
                                        step=step)
                    elif isinstance(metric_v, str):
                        mlflow.log_text(metric_v, "{}_eval/{}_report.txt".format(split,step))
        
        if split == "valid":

            if metrics[f"{split}-{self.default_metric}"] >= self.best_metric:
                self.best_metric = metrics[f"{split}-{self.default_metric}"]

                self.best_model_path = Path("{}/model_step_{}_{}_{}".format(
                    self.best_model_dir,self.global_valid_eval_epoch, self.default_metric, round(metrics[f"{split}-{self.default_metric}"], 3)))
                
                if self.global_rank == 0:
                    
                    if self.best_model_path.exists():
                        shutil.rmtree(self.best_model_path)
                    
                    mlflow.pytorch.save_model(
                        (self.net.module if is_parallel(self.net) else self.net),
                        self.best_model_path,
                        code_paths=[os.path.join(self.root_level_dir, "models")])
                # ipdb.set_trace()
        return metrics        

        
    def train_epoch(self):

        self.net.train()

        for _, data in enumerate(self.dataloaders['train']):

            sequence, name2id, label = data
            
            token_sequence = self.feature_fetcher.query_features(sequence['peptide'])
            
            batch = {k: torch.tensor(v).to(self.device) for k, v in token_sequence.items()}
            
            labels = torch.tensor(label[self.cfg.task.type]).to(self.device)
                
            output = self.net(batch)
            
            loss = self.reg_loss_func(output[0].squeeze(),labels.squeeze())

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.cfg.train.lr_scheduler.when == "batch" and self.cfg.train.lr_scheduler.type in ("onecycle", "cosine"):
                self.scheduler.step()

            # logging

            if self.global_rank == 0 and (self.global_train_step +
                    1) % self.cfg.logger.log_per_steps == 0:

                cur_lr = self.scheduler.optimizer.state_dict(
                )['param_groups'][0]['lr']

                Logger.info("lr: {:.8f}".format(cur_lr))

                # for k, v in loss_dict.items():
                
                Logger.info(
                    "train | epoch: {:d}  step: {:d} | loss: {:.4f}".
                    format(self.epoch, self.global_train_step,loss.item()))

                if not self.cfg.mode.nni and self.global_rank == 0 and self.cfg.logger.log:
                    mlflow.log_metric("lr",
                                      float(cur_lr),
                                      step=self.global_train_step)
                    # for k, v in loss_dict.items():
                    mlflow.log_metric("train/loss",
                                    loss.item(),
                                    step=self.global_train_step)
            
            self.global_train_step += 1

    def run(self):
        
        self.eval_epoch('train')

        for epoch in range(1, self.num_epoch+1):

            
            
            self.epoch = epoch
            
            if self.cfg.mode.ddp:
                self.dataloaders["train"].sampler.set_epoch(self.epoch)
            
            # if not self.cfg.mode.full:
            if self.global_rank == 0 and self.epoch % self.cfg.train.eval_epoch == 0:
                train_metrics = self.eval_epoch('train')
                valid_metrics = self.eval_epoch("valid")    

                # learning rate scheduler
                if self.cfg.train.lr_scheduler.when == "epoch":
                    if self.cfg.train.lr_scheduler.type in ("plateau",):
                        self.scheduler.step(valid_metrics[f"valid_{self.default_metric}"])
                    elif self.cfg.train.lr_scheduler.type in ("onecycle", "cosine"):
                        self.scheduler.step()

            start = time.time()
            self.train_epoch()
            end = time.time()     

            Logger.info(f'epoch {epoch}:cost {end-start}s')

            
