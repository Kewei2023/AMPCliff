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
from AMPCliff.utils.evaluator import Evaluator #######
from AMPCliff.features.feature_fetcher import FeatureFetcher
from AMPCliff.factory.initializer import ModelInitializer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from AMPCliff.utils.metrics import Metrics
from AMPCliff.visualization.plot import plot_low_dimension
from AMPCliff.utils.match import seq2pair
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

   
        
    for condition in cfg.data[cfg.task.type].condition:
          
      for diff in cfg.data.diff: 
      
        if cfg.data[cfg.task.type].mode == 'fix':
    
            # diff = cfg.data.diff
            if global_rank == 0:
                Logger.info('loading train data...')
            
            train_file_path = cfg.data[cfg.task.type].fix.train_file.replace("{diff}",str(diff)).replace("{condition}",condition)
              
            
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
            
            valid_file_path = cfg.data[cfg.task.type].fix.valid_file.replace("{diff}",str(diff)).replace("{condition}",condition)
              
            
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
            
            test_file_path = cfg.data[cfg.task.type].fix.test_file.replace("{diff}",str(diff)).replace("{condition}",condition)
              
            
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
            
            if global_rank == 0:
                Logger.info('loading checkpoint...')
                
            if not cfg.model[cfg.task.type].check_point.load:
              
              Logger.info('please give a check point path and set "load" to true')
              exit()
              
            else:
              model = load_model(model, cfg.model[cfg.task.type].check_point.path, device)
           
            
            evaluator = Evaluator(model,dataloaders, metric_func,feature_fetcher, device, cfg)
             
            
            if global_rank == 0:
              Logger.info("evaluating train set......")
            
            evaluate_tr_metrics = evaluator.run('train')
            y_pred,y_true,ids = evaluator.y_pred,evaluator.y_true,evaluator.ids
              
            if cfg.mode.ddp:
              # collect results of each gpu
              gather_list_pred = [torch.zeros_like(y_pred) for _ in range(world_size)]
              gather_list_true = [torch.zeros_like(y_true) for _ in range(world_size)]
              gather_list_ids = [torch.zeros_like(ids) for _ in range(world_size)]
              
              torch.distributed.all_gather(gather_list_pred, y_pred)
              torch.distributed.all_gather(gather_list_true, y_true)
              torch.distributed.all_gather(gather_list_ids, ids)
            
              if global_rank == 0:
                # concat all results
                y_pred = torch.cat(gather_list_pred)
                y_true = torch.cat(gather_list_true)
                ids = torch.cat(gather_list_ids)
            
            
            if global_rank == 0:
              # plot figures
              metric_func(y_pred,y_true ,split='train',plot=True)
              # save results
              df = pd.read_csv(train_file_path)
                  
              pred_df = pd.DataFrame.from_dict({f'{cfg.model[cfg.task.type].version}-pred':y_pred.squeeze(1).cpu(),
                                                'true':y_true.cpu(),
                                                'Idx':ids})
              res_df = pd.merge(df,pred_df,on='Idx',how='outer').sort_values(by='Idx')
              res_df.to_csv(f'./{cfg.model[cfg.task.type].version}-{condition}-diff{diff}-train_result.csv')
            
            
            
            if global_rank == 0:
              Logger.info("evaluating valid set......")
              
            evaluate_val_metrics = evaluator.run('valid')
            y_pred,y_true,ids = evaluator.y_pred,evaluator.y_true,evaluator.ids
            
            
            if cfg.mode.ddp:
              # collect results of each gpu
              gather_list_pred = [torch.zeros_like(y_pred) for _ in range(world_size)]
              gather_list_true = [torch.zeros_like(y_true) for _ in range(world_size)]
              gather_list_ids = [torch.zeros_like(ids) for _ in range(world_size)]
              
              torch.distributed.all_gather(gather_list_pred, y_pred)
              torch.distributed.all_gather(gather_list_true, y_true)
              torch.distributed.all_gather(gather_list_ids, ids)
            
              if global_rank == 0:
                # concat all results
                y_pred = torch.cat(gather_list_pred)
                y_true = torch.cat(gather_list_true)
                ids = torch.cat(gather_list_ids)
                
            if global_rank == 0: 
            
              metric_func(y_pred,y_true ,split='valid',plot=True)
              # save results
              df = pd.read_csv(valid_file_path)
                  
              pred_df = pd.DataFrame.from_dict({f'{cfg.model[cfg.task.type].version}-pred':y_pred.squeeze(1).cpu(),
                                                'true':y_true.cpu(),
                                                'Idx':ids})
              res_df = pd.merge(df,pred_df,on='Idx',how='outer').sort_values(by='Idx')
              res_df.to_csv(f'./{cfg.model[cfg.task.type].version}-{condition}-diff{diff}-valid_result.csv')
            
            
            
            
            if global_rank == 0:
              Logger.info("evaluating test set......")
              Logger.info("and save pair info......")
              
            evaluate_test_metrics = evaluator.run('test')
            y_pred,y_true,ids = evaluator.y_pred,evaluator.y_true,evaluator.ids
            
            if cfg.mode.ddp:
              # collect results of each gpu
              gather_list_pred = [torch.zeros_like(y_pred) for _ in range(world_size)]
              gather_list_true = [torch.zeros_like(y_true) for _ in range(world_size)]
              gather_list_ids = [torch.zeros_like(ids) for _ in range(world_size)]
              
              torch.distributed.all_gather(gather_list_pred, y_pred)
              torch.distributed.all_gather(gather_list_true, y_true)
              torch.distributed.all_gather(gather_list_ids, ids)
            
              if global_rank == 0:
                # concat all results
                y_pred = torch.cat(gather_list_pred)
                y_true = torch.cat(gather_list_true)
                ids = torch.cat(gather_list_ids)
                
            if global_rank == 0: 
            
              # plot figures
              metric_func(y_pred,y_true ,split='test',plot=True)
              # save results
              df = pd.read_csv(test_file_path)
                  
              pred_df = pd.DataFrame.from_dict({f'{cfg.model[cfg.task.type].version}-pred':y_pred.squeeze(1).cpu(),
                                                'true':y_true.cpu(),
                                                'Idx':ids})
              res_df = pd.merge(df,pred_df,on='Idx',how='outer').sort_values(by='Idx')
              res_df.to_csv(f'./{cfg.model[cfg.task.type].version}-{condition}-diff{diff}-test_result.csv')
            
            
            
            if global_rank == 0:
                if not cfg.mode.nni and cfg.logger.log:            
                    for metric_name, metric_v in evaluate_tr_metrics.items():
                        if isinstance(metric_v, (float, np.float64, int, np.int)):
                            mlflow.log_metric("train_final/{}".format(metric_name), metric_v, step=1)
                        elif isinstance(metric_v, str):
                            mlflow.log_text(metric_v, "train_final/report.txt")
                    
                    for metric_name, metric_v in evaluate_val_metrics.items():
                        if isinstance(metric_v, (float, np.float64, int, np.int)):
                            mlflow.log_metric("valid_final/{}".format(metric_name), metric_v, step=1)
                        elif isinstance(metric_v, str):
                            mlflow.log_text(metric_v, "valid_final/report.txt")
                    
                    for metric_name, metric_v in evaluate_test_metrics.items():
                        if isinstance(metric_v, (float, np.float64, int, np.int)):
                            mlflow.log_metric("test_final/{}".format(metric_name), metric_v, step=1)
                        elif isinstance(metric_v, str):
                            mlflow.log_text(metric_v, "test_final/report.txt")



    if cfg.mode.ddp:
        cleanup_multinodes()
      
if __name__ == '__main__':
    main()
    print('done.')
