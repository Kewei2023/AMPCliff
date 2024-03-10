

import torch
# import nni
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from torch.nn import MSELoss
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
import time
from .loss import RankLoss
import numpy as np

class Trainer(object):

    def __init__(self, net, dataloaders, optimizer, scheduler, metrics,feature_fetcher,
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
        # self.rank_loss_func = RankLoss(margin=np.log10(5))
        self.reg_loss_func = MSELoss()
        # metrics
        self.metrics_func = metrics

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
            y_pred_1, y_true_1 = [],[]
            y_pred_2, y_true_2 = [],[]
            y_pred_12, y_true_12 = [],[]
            # latent = []
            # metrics = {}
            for step, data in tqdm(
                    enumerate(self.dataloaders[split]),
                    total=len(self.dataloaders[split]),
                    desc="evaluating {}| loss: {:.4f}".format(split,loss)):

                sequence1, sequence2, pairids, label= data
                # ipdb.set_trace()
                token_sequence1 = self.feature_fetcher.query_features(sequence1['peptide'])
                token_sequence2 = self.feature_fetcher.query_features(sequence2['peptide'])
                
                batch1 = {k: torch.tensor(v).to(self.device) for k, v in token_sequence1.items()}
                   
                batch2 = {k: torch.tensor(v).to(self.device) for k, v in token_sequence2.items()}
                
                regression_output1, regression_output2,regression_output12 = self.net(batch1,batch2)
                # ipdb.set_trace()
                # rank_loss = self.rank_loss_func(rank_output1, rank_output2)
                reg_loss1 = self.reg_loss_func(regression_output1.squeeze().cpu(),torch.tensor(label['regression1']).squeeze())
                reg_loss2 = self.reg_loss_func(regression_output2.squeeze().cpu(),torch.tensor(label['regression2']).squeeze())
                reg_loss12 = self.reg_loss_func(regression_output12.squeeze().cpu(),torch.tensor(label['regression12']).squeeze())
                
                if self.cfg.train.loss_type == '12+1+2':
                    loss = reg_loss12 + reg_loss1 + reg_loss2
                if self.cfg.train.loss_type == '1+2':   
                    loss = reg_loss1 + reg_loss2
                if self.cfg.train.loss_type == '12':   
                    loss = reg_loss12
                

                y_pred_1.append(regression_output1)
                y_true_1.append(torch.tensor(label['regression1']))

                y_pred_2.append(regression_output2)
                y_true_2.append(torch.tensor(label['regression2']))

                y_pred_12.append(regression_output12)
                y_true_12.append(torch.tensor(label['regression12']))
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
            y_pred_1 = torch.cat(y_pred_1)
            y_true_1 = torch.cat(y_true_1)
            y_pred_2 = torch.cat(y_pred_2)
            y_true_2 = torch.cat(y_true_2)
            y_pred_12 = torch.cat(y_pred_12)
            y_true_12 = torch.cat(y_true_12)
            
        metrics1 = self.metrics_func(y_pred_1, y_true_1,split)
        metrics2 = self.metrics_func(y_pred_2, y_true_2,split)
        metrics12 = self.metrics_func(y_pred_12, y_true_12,split)
        # ipdb.set_trace()
        
        res_dict = {}
        # for k,v in loss_dict.items():
        #     res_dict["{}_{}".format(split, k )] = np.mean(v)
        # res_dict['loss'] = np.mean(total_loss)
        for k in metrics1:
            res_dict[f"{split}-{k}"] = (metrics1[k]+metrics2[k])/2
            res_dict[f"{split}-{k}-1"] = metrics1[k]
            res_dict[f"{split}-{k}-2"] = metrics2[k]
            res_dict[f"{split}-{k}-12"] = metrics12[k]
            
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
                    if isinstance(metric_v,  (float, np.float, int, np.int_)):
                        mlflow.log_metric("{}_eval/{}".format(split,metric_name),
                                        metric_v,
                                        step=step)
                    elif isinstance(metric_v, str):
                        mlflow.log_text(metric_v, "{}_eval/{}_report.txt".format(split,step))
        
        if split == "valid":

            if metrics[f"{split}-{self.default_metric}"] > self.best_metric:
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

           
            token_sequence1,token_sequence2, pairids, label= data
            # ipdb.set_trace()
            batch1 = {k: torch.tensor(v).to(self.device) for k, v in token_sequence1.items()}
                
            batch2 = {k: torch.tensor(v).to(self.device) for k, v in token_sequence2.items()}
            
            regression_output1, regression_output2,regression_output12 = self.net(batch1,batch2)
                
            reg_loss1 = self.reg_loss_func(regression_output1.squeeze().cpu(),torch.tensor(label['regression1']).squeeze())
            reg_loss2 = self.reg_loss_func(regression_output2.squeeze().cpu(),torch.tensor(label['regression2']).squeeze())
            reg_loss12 = self.reg_loss_func(regression_output12.squeeze().cpu(),torch.tensor(label['regression12']).squeeze())
            


            if self.cfg.train.loss_type == '12+1+2':
                loss = reg_loss12 + reg_loss1 + reg_loss2
            if self.cfg.train.loss_type == '1+2':   
                loss = reg_loss1 + reg_loss2
            if self.cfg.train.loss_type == '12':   
                loss = reg_loss12
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
                    "train | epoch: {:d}  step: {:d} | loss: {:.4f} | reg loss12: {:.4f} | reg loss1: {:.4f} | reg loss2: {:.4f}".
                    format(self.epoch, self.global_train_step,loss.item(),reg_loss12.item(),reg_loss1.item(),reg_loss2.item()))

                if not self.cfg.mode.nni and self.global_rank == 0 and self.cfg.logger.log:
                    mlflow.log_metric("lr",
                                      float(cur_lr),
                                      step=self.global_train_step)
                    # for k, v in loss_dict.items():
                    mlflow.log_metric("train/loss",
                                    loss.item(),
                                    step=self.global_train_step)
                    mlflow.log_metric("train/reg_loss12",
                                    reg_loss12.item(),
                                    step=self.global_train_step)
                    mlflow.log_metric("train/reg_loss1",
                                    reg_loss1.item(),
                                    step=self.global_train_step)
                    mlflow.log_metric("train/reg_loss2",
                                    reg_loss2.item(),
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

            
