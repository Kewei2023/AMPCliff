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
from AMPCliff.loader.utils_test import make_loader #######
from AMPCliff.loader.split import random_split_data,stratified_split_data,fixed_cluster_split_data
from AMPCliff.utils.utils import get_device,fix_random_seed, load_weights, load_model
from AMPCliff.models.breeze import BreezeModel,BreezeTokenizer,BreezeForMaskedLM,BreezeForSequenceClassification, BreezeConfig
from AMPCliff.utils.trainer_test import Trainer ######
from AMPCliff.utils.evaluator_test import Evaluator #######
from AMPCliff.features.feature_fetcher import FeatureFetcher
from AMPCliff.utils.trainer_pair import Trainer as PTrainer
from AMPCliff.utils.evaluator_pair import Evaluator as PEvaluator
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

        
        
        if global_rank == 0:
            Logger.info('loading train data...')
        
        train_dataloader = make_loader(
                                        local_rank=local_rank,
                                        dataset_file=cfg.data[cfg.task.type].fix.train_file,
                                        cfg = cfg,
                                        batch_size=cfg.train.batch_size,
                                        vocab_dict=vocab_dict,
                                        pin_memory=False,
                                        num_workers=cfg.train.num_workers,
                                        random_seed=random_seed
                                    )
        if global_rank == 0:
            Logger.info('loading valid data...')
        valid_dataloader = make_loader(
                                        local_rank=local_rank,
                                        dataset_file=cfg.data[cfg.task.type].fix.valid_file,
                                        cfg = cfg,
                                        batch_size=cfg.train.batch_size,
                                        vocab_dict=vocab_dict,
                                        pin_memory=False,
                                        num_workers=cfg.train.num_workers,
                                        random_seed=random_seed
                                    )
        if global_rank == 0:
            Logger.info('loading test data...')
        test_dataloader = make_loader(
                                        local_rank=local_rank,
                                        dataset_file=cfg.data[cfg.task.type].fix.test_file,
                                        cfg = cfg,
                                        batch_size=cfg.train.batch_size,
                                        vocab_dict=vocab_dict,
                                        pin_memory=False,
                                        num_workers=cfg.train.num_workers,
                                        random_seed=random_seed
                                    )

        dataloaders = {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}
        
        
        if cfg.task.type =='rank':
            evaluator = PEvaluator(model,dataloaders, metric_func,feature_fetcher, device, cfg)
        
        else:
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
        
        
        

    # 5 fold evaluations
    if cfg.data[cfg.task.type].mode == 'random':
        
        if cfg.data[cfg.task.type].random.stratified:
            stratified_split_data(data_path = cfg.data[cfg.task.type].random.data_file, 
                                fold = cfg.data[cfg.task.type].random.fold,
                                output_dir = cfg.data[cfg.task.type].random.fold_files)
            # if not os.path.exists(os.path.join(cfg.data[cfg.task.type].random.fold_files,'train_fold_0.csv')):
        else:
            random_split_data(data_path = cfg.data[cfg.task.type].random.data_file, 
                            fold = cfg.data[cfg.task.type].random.fold,
                            output_dir = cfg.data[cfg.task.type].random.fold_files)
    
    if cfg.data[cfg.task.type].mode == 'fixed_cluster':
        # ipdb.set_trace()
        fixed_cluster_split_data(data_path = cfg.data[cfg.task.type].fixed_cluster.data_file, 
                        fold = cfg.data[cfg.task.type].fixed_cluster.fold,
                        output_dir = cfg.data[cfg.task.type].fixed_cluster.fold_files)
    
        # exit()   
        models = {}

        all_y_true, all_y_pred = [], []
        results_df = []
        all_latent = []

        result_dir = 'folds'
        os.makedirs(result_dir,exist_ok=True)

        for fold in range(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold):
            
            
            model, tokenizer = ModelInitializer(cfg,device).init()

            optimizer = AdamW(model.parameters(),
                lr=cfg.train.learning_rate,
                betas=(0.9, 0.999),
                eps=cfg.train.adam_epsilon,
                weight_decay=cfg.train.weight_decay)

            scheduler = lr_scheduler(cfg, optimizer)
            
            if global_rank == 0:
                Logger.info('loading train data...')
            
            train_dataloader = make_loader(
                                            local_rank=local_rank,
                                            dataset_file=os.path.join(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold_files,f'train_fold_{fold}.csv'),
                                            cfg = cfg,
                                            batch_size=cfg.train.batch_size,
                                            vocab_dict=vocab_dict,
                                            pin_memory=False,
                                            num_workers=cfg.train.num_workers,
                                            random_seed=random_seed
                                        )
            # ipdb.set_trace()
            if global_rank == 0:
                Logger.info('loading valid data...')
            valid_dataloader = make_loader(
                                            local_rank=local_rank,
                                            dataset_file=os.path.join(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold_files,f'test_fold_{fold}.csv'),
                                            cfg = cfg,
                                            batch_size=cfg.train.batch_size,
                                            vocab_dict=vocab_dict,
                                            pin_memory=False,
                                            num_workers=cfg.train.num_workers,
                                            random_seed=random_seed
                                        )

            dataloaders = {'train': train_dataloader, 'valid': valid_dataloader}
            
            if len(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].checkpoints) != cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold:
                if cfg.task.type == 'rank':
                    trainer = PTrainer(model, dataloaders, optimizer, scheduler, metric_func,feature_fetcher,
                                device, global_rank, cfg,f'fold {fold}')
                    
                else:
                    trainer = Trainer(model, dataloaders, optimizer, scheduler, metric_func,feature_fetcher,
                                device, global_rank, cfg,f'fold {fold}')
                
                trainer.run()

                if global_rank == 0:
                    
                    Logger.info(f"finished training fold {fold}......")
                    Logger.info("start evaluating......")
                    Logger.info("loading best weights from {}......".format(
                        trainer.best_model_path))
                    
                    model = load_weights(model, trainer.best_model_path, device)
                    # models[fold] = model
            else:
                model = load_model(model, cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].checkpoints[fold], device)
                # model = load_weights(model, trainer.best_model_path, device)
            if cfg.task.type =='rank':
                evaluator = PEvaluator(model,dataloaders, metric_func,feature_fetcher, device, cfg)
            
            else:
                evaluator = Evaluator(model,dataloaders, metric_func,feature_fetcher, device, cfg)
            
            # save valid result
            evaluator.run('valid')
            
            y_pred,y_true,ids = evaluator.y_pred,evaluator.y_true,evaluator.ids# ,evaluator.names,evaluator.ids,evaluator.latent
            
            if cfg.task.type =='rank':
                all_y_pred.append(torch.cat(y_pred))
                all_y_true.append(torch.cat(y_true))
            else:
                all_y_pred.append(y_pred)
                all_y_true.append(y_true)
            # all_latent.append(latent)

            df = pd.read_csv(os.path.join(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold_files,f'test_fold_{fold}.csv'))
            
            if len(y_pred)==2:
                pred_df = pd.DataFrame.from_dict(dict(pred1=y_pred[0].squeeze(1).cpu(),true1=y_true[0].cpu(),pred2=y_pred[1].squeeze(1).cpu(),true2=y_true[1].cpu(),Idx=ids))
            else:
                pred_df = pd.DataFrame.from_dict(dict(pred=y_pred.squeeze(1).cpu(),true=y_true.cpu(),Idx=ids))
            
            # ipdb.set_trace()
            res_df = pd.merge(df,pred_df,on='Idx',how='outer')

            res_df.to_csv(f'{result_dir}/test_fold_{fold}.csv')
            # ipdb.set_trace()
            results_df.append(res_df)

            # save train result
            evaluator.run('train')
            y_pred,y_true,ids = evaluator.y_pred,evaluator.y_true,evaluator.ids# ,evaluator.latent
            
            df = pd.read_csv(os.path.join(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold_files,f'train_fold_{fold}.csv'))
            
            if len(y_pred)==2:
                pred_df = pd.DataFrame.from_dict(dict(pred1=y_pred[0].squeeze(1).cpu(),true1=y_true[0].cpu(),pred2=y_pred[1].squeeze(1).cpu(),true2=y_true[1].cpu(),Idx=ids))
            else:
                pred_df = pd.DataFrame.from_dict(dict(pred=y_pred.squeeze(1).cpu(),true=y_true.cpu(),Idx=ids))
            
            res_df = pd.merge(df,pred_df,on='Idx',how='outer')

            res_df.to_csv(f'{result_dir}/train_fold_{fold}.csv')

        # ipdb.set_trace()
        all_y_pred = torch.cat(all_y_pred)
        all_y_true = torch.cat(all_y_true)
        # all_latent = torch.cat(all_latent).mean(1)
        # ipdb.set_trace()

        # save results
        final_results = pd.concat(results_df,0)

        # label_quality_scores = get_label_quality_scores(labels=all_y_true, predictions=all_y_pred)
        # final_results['is_label_issue'] = label_quality_scores
        final_results.to_csv('final_results.csv')

        # plot scatter
        metrics = metric_func(all_y_pred, all_y_true,split='final',plot=True)

        # plot the last hidden_states
        # plot_low_dimension(all_latent, labels=final_results['group'])

        
        if global_rank == 0:
            if not cfg.mode.nni and cfg.logger.log:            
                for metric_name, metric_v in metrics.items():
                    if isinstance(metric_v, (float, np.float64, int)):
                        mlflow.log_metric("cv_final/{}".format(metric_name), metric_v, step=1)
                    elif isinstance(metric_v, str):
                        mlflow.log_text(metric_v, "cv_final/report.txt")


    if cfg.mode.ddp:
        cleanup_multinodes()
      
if __name__ == '__main__':
    main()
    print('done.')
