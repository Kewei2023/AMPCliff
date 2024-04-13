# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import pandas as pd
import hydra
from pathlib import Path
from omegaconf import DictConfig
from AMPCliff.utils.std_logger import Logger
from AMPCliff.utils.scheduler import lr_scheduler
from AMPCliff.utils.distribution import setup_multinodes, cleanup_multinodes
from AMPCliff.loader.utils import make_loader #######
from AMPCliff.loader.split import random_split_data,stratified_split_data,fixed_cluster_split_data
from AMPCliff.utils.utils import get_device,fix_random_seed, load_weights, load_model
from AMPCliff.models.breeze import BreezeModel,BreezeTokenizer,BreezeForMaskedLM,BreezeForSequenceClassification, BreezeConfig
from AMPCliff.utils.trainer import Trainer ######
from AMPCliff.utils.evaluator import Evaluator #######
from AMPCliff.features.feature_fetcher import FeatureFetcher
from AMPCliff.factory.initializer import ModelInitializer
from transformers import EsmModel,EsmForSequenceClassification,LlamaForCausalLM, LlamaTokenizer, EsmForMaskedLM
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from AMPCliff.utils.metrics import Metrics
from AMPCliff.visualization.plot import plot_low_dimension
from cleanlab.regression.rank import get_label_quality_scores
import mlflow
import torch
import ipdb
# from progen2.models.progen.modeling_progen import ProGenModel,ProGenForCausalLM



########################################################################
# util

@hydra.main(config_path="configs", config_name="downstream.yaml")
def main(cfg: DictConfig):
    
    orig_cwd = hydra.utils.get_original_cwd()
    cfg.orig_cwd = orig_cwd
    local_rank = 0
    global_rank = 0
    # ipdb.set_trace()
    if not cfg.mode.nni and cfg.logger.log:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])


    if cfg.mode.ddp:

        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        global_rank = int(os.environ['RANK'])
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_SHM_DISABLE'] = '1'
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
        random_seed = cfg.train.random_seed + local_rank
        setup_multinodes(local_rank, world_size)
        device = torch.device("cuda", local_rank)
        # deepspeed.init_distributed()
    else:
        random_seed = cfg.train.random_seed
        device = get_device(cfg)

    if global_rank == 0:
        Logger.info("setting random seed: {}".format(random_seed))

    fix_random_seed(random_seed, cuda_deterministic=True)
    
    if cfg.logger.log and global_rank == 0:

        # log hyper-parameters
        for p, v in cfg.mode.items():
            mlflow.log_param(p, v)
            
        for p, v in cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].items():
           
            mlflow.log_param(p, v)

        for p, v in cfg.train.items():
            mlflow.log_param(p, v)

        for p, v in cfg.model[cfg.task.type].items():
           mlflow.log_param(p, v)



    metric_func = Metrics(cfg.task.type,topK=50)
    model, tokenizer = ModelInitializer(cfg,device).init()
    
    vocab_dict = tokenizer.get_vocab()
    
    feature_fetcher = FeatureFetcher(cfg,tokenizer)
    
    if cfg.mode.ddp:
        model = DDP(model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True
                    )

    # fix data files
    if cfg.data[cfg.task.type].mode == 'fix':

        
        diff = cfg.data.diff
        
        if global_rank == 0:
            Logger.info('loading train data...')
        
        train_file_path = cfg.data[cfg.task.type].fix.train_file.replace("{diff}",str(diff)).replace("{condition}",cfg.data[cfg.task.type].condition)
        
        train_dataloader = make_loader(
                                        local_rank=local_rank,
                                        dataset_file=train_file_path,
                                        cfg = cfg,
                                        batch_size=cfg.train.batch_size,
                                        vocab_dict=vocab_dict,
                                        pin_memory=False,
                                        num_workers=cfg.train.num_workers,
                                        random_seed=random_seed
                                    )
        if global_rank == 0:
            Logger.info('loading valid data...')
        
        valid_file_path = cfg.data[cfg.task.type].fix.valid_file.replace("{diff}",str(diff)).replace("{condition}",cfg.data[cfg.task.type].condition)
        
        valid_dataloader = make_loader(
                                        local_rank=local_rank,
                                        dataset_file=valid_file_path,
                                        cfg = cfg,
                                        batch_size=cfg.train.batch_size,
                                        vocab_dict=vocab_dict,
                                        pin_memory=False,
                                        num_workers=cfg.train.num_workers,
                                        random_seed=random_seed
                                    )
        if global_rank == 0:
            Logger.info('loading test data...')
        
        test_file_path = cfg.data[cfg.task.type].fix.test_file.replace("{diff}",str(diff)).replace("{condition}",cfg.data[cfg.task.type].condition)
        
        test_dataloader = make_loader(
                                        local_rank=local_rank,
                                        dataset_file=test_file_path,
                                        cfg = cfg,
                                        batch_size=cfg.train.batch_size,
                                        vocab_dict=vocab_dict,
                                        pin_memory=False,
                                        num_workers=cfg.train.num_workers,
                                        random_seed=random_seed
                                    )
                                    
        dataloaders = {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}
        
        
        
        evaluator = Evaluator(model,dataloaders, metric_func,feature_fetcher, device, cfg)
         
        all_latent = []
        all_dataset = []
        evaluate_tr_metrics = evaluator.run('train')
        y_pred,y_true,latent = evaluator.y_pred,evaluator.y_true,evaluator.latent
        metric_func(y_pred,y_true ,split='train',plot=True)
        
        all_latent.append(latent)
        all_dataset.extend(['train']*latent.shape[0])
        
        evaluate_val_metrics = evaluator.run('valid')
        y_pred,y_true,latent  = evaluator.y_pred,evaluator.y_true,evaluator.latent
        metric_func(y_pred,y_true ,split='valid',plot=True)
        
        all_latent.append(latent)
        all_dataset.extend(['valid']*latent.shape[0])
        
        evaluate_test_metrics = evaluator.run('test')
        y_pred,y_true,latent  = evaluator.y_pred,evaluator.y_true,evaluator.latent
        metric_func(y_pred,y_true ,split='test',plot=True)
        
        all_latent.append(latent)
        all_dataset.extend(['test']*latent.shape[0])
        
        plot_low_dimension(torch.cat(all_latent,0),labels=np.array(all_dataset),savedir='visual_latent',alpha=0.3)
    
    if cfg.mode.ddp:
        cleanup_multinodes()
      
if __name__ == '__main__':
    main()
    print('done.')
