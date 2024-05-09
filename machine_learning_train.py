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
from AMPCliff.loader.utils import make_loader, load_raw_seq #######
from AMPCliff.loader.split import random_split_data,stratified_split_data,fixed_cluster_split_data
from AMPCliff.utils.utils import get_device,fix_random_seed
from AMPCliff.features.feature_fetcher import FeatureFetcher
from AMPCliff.factory.ML import ModelRegressor
from AMPCliff.utils.metrics import Metrics
from AMPCliff.visualization.plot import plot_low_dimension
from AMPCliff.utils.match_ml import seq2pair
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler
from functools import reduce
import mlflow
import torch
import ipdb
# from progen2.models.progen.modeling_progen import ProGenModel,ProGenForCausalLM



########################################################################
# util

@hydra.main(config_path="configs", config_name="ML.yaml")
def main(cfg: DictConfig):
    
    orig_cwd = hydra.utils.get_original_cwd()
    cfg.orig_cwd = orig_cwd
    
    random_seed = cfg.train.random_seed
       
    if not cfg.mode.nni and cfg.logger.log:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
        
    fix_random_seed(random_seed, cuda_deterministic=True)
    
    if cfg.logger.log:

        # log hyper-parameters
        for p, v in cfg.mode.items():
            mlflow.log_param(p, v)
            
        for p, v in cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].items():
           
            mlflow.log_param(p, v)

        for p, v in cfg.train.items():
            mlflow.log_param(p, v)

        


    metric_func = Metrics(cfg.task.type,topK=50)
  
    feature_fetcher = FeatureFetcher(cfg,tokenizer=None)
    
    models = ModelRegressor(random_state=random_seed)
    
    scaler = StandardScaler()
    
    if cfg.data[cfg.task.type].mode == 'fix':
      for condition in cfg.data[cfg.task.type].condition:
      
        for diff in cfg.data.diff: # [2,3,4,5]:
          
          Logger.info(f'testing the diff {diff} data')
          savedir = f'./{condition}/diff{diff}'
          os.makedirs(savedir,exist_ok=True)
          
          if cfg.data[cfg.task.type].mode == 'fix':
            
            
            train_file_path = cfg.data[cfg.task.type].fix.all_train_file.replace("{diff}",str(diff)).replace("{condition}",condition)
            train_sequence, train_seqName, train_seqID, train_label_dict = load_raw_seq(dataset_file=train_file_path,
                  vocab_dict=None,
                  cfg=cfg)
            
            test_file_path = cfg.data[cfg.task.type].fix.test_file.replace("{diff}",str(diff)).replace("{condition}",condition)
            valid_sequence, valid_seqName, valid_seqID, valid_label_dict = load_raw_seq(dataset_file=test_file_path,
            vocab_dict=None,
            cfg=cfg)
            '''      
            train_sequence, train_seqName, train_seqID, train_label_dict = load_raw_seq(dataset_file=cfg.data[cfg.task.type].fix[f"diff{diff}"].all_train_file,
                  vocab_dict=None,
                  cfg=cfg)
                  
            valid_sequence, valid_seqName, valid_seqID, valid_label_dict = load_raw_seq(dataset_file=cfg.data[cfg.task.type].fix[f"diff{diff}"].test_file,
            vocab_dict=None,
            cfg=cfg)
            ''' 
            
            # scale the features
                  
            x_train = scaler.fit_transform(feature_fetcher.query_features(train_sequence)['x'])
            x_test = scaler.transform(feature_fetcher.query_features(valid_sequence)['x'])
            
            # training
            models.train( X_train = x_train, 
                          y_train = train_label_dict['regression'])
            
            train_result_dict = models.predict(X_test = x_train)
                                      
            valid_result_dict = models.predict(X_test = x_test)
                        
            
            df = pd.read_csv(test_file_path,index_col=0)
            # df = pd.read_csv(cfg.data[cfg.task.type].fix[f"diff{diff}"].test_file)
      
            pred_df = pd.DataFrame(valid_result_dict, index = valid_seqName).sort_index()
            
            # ipdb.set_trace()
            res_df = pd.concat([df,pred_df], axis=1)
            
            
            res_df.to_csv(os.path.join(savedir,'test_result.csv'))
            
            if cfg.data[cfg.task.type].acindex:
              pair_df = seq2pair(res_df,diff)
              pair_df.to_csv(os.path.join(savedir,'test_result_pair.csv'))
            
            # 
            
      
            # save train result           
            df = pd.read_csv(train_file_path,index_col=0)
            
            pred_df = pd.DataFrame(train_result_dict, index = train_seqName).sort_index()
            
            res_df = pd.concat([df,pred_df], axis=1)
      
            res_df.to_csv(os.path.join(savedir,'train_result.csv'))
            
            for model in train_result_dict:
                # plot scatter
                metrics = metric_func(train_result_dict[model], np.array(train_label_dict['regression']),split=f'{savedir}/train-{model}',plot=True)
                metrics = metric_func(valid_result_dict[model], np.array(valid_label_dict['regression']),split=f'{savedir}/test-{model}',plot=True)
      
      
      
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
           
        

        
        results_df = []
        
        savedir = 'folds'
        os.makedirs(savedir,exist_ok=True)
                           
        for fold in range(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold): 
          
          print("*"*30)
          print(f"FOLD {fold}")
          print("*"*30)          
                       
          train_file_path = os.path.join(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold_files,f'train_fold_{fold}.csv')
          
          train_sequence, train_seqName, train_seqID, train_label_dict = load_raw_seq(dataset_file=train_file_path,
                vocab_dict=None,
                cfg=cfg)
          
          test_file_path = os.path.join(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold_files,f'test_fold_{fold}.csv')
          valid_sequence, valid_seqName, valid_seqID, valid_label_dict = load_raw_seq(dataset_file=test_file_path,
          vocab_dict=None,
          cfg=cfg)
          
          
          # scale the features
                
          x_train = scaler.fit_transform(feature_fetcher.query_features(train_sequence)['x'])
          x_test = scaler.transform(feature_fetcher.query_features(valid_sequence)['x'])
          
          # training
          models.train( X_train = x_train, 
                        y_train = train_label_dict['regression'])
          
          train_result_dict = models.predict(X_test = x_train)
                                    
          valid_result_dict = models.predict(X_test = x_test)
                      
          
          df = pd.read_csv(test_file_path,index_col=0)
          # df = pd.read_csv(cfg.data[cfg.task.type].fix[f"diff{diff}"].test_file)
    
          pred_df = pd.DataFrame(valid_result_dict, index = valid_seqName)
          
          ipdb.set_trace()
          res_df = pd.concat([df,pred_df], axis=1)
          
          
          res_df.to_csv(os.path.join(savedir,f'test_result_fold{fold}.csv'))
          
          results_df.append(res_df)
          if cfg.data[cfg.task.type].acindex:
            pair_df = seq2pair(res_df,diff)
            pair_df.to_csv(os.path.join(savedir,'test_result_pair.csv'))
          
          # 
          
    
          # save train result           
          df = pd.read_csv(train_file_path,index_col=0)
          
          pred_df = pd.DataFrame(train_result_dict, index = train_seqName).sort_index()
          
          res_df = pd.concat([df,pred_df], axis=1)
    
          res_df.to_csv(os.path.join(savedir,f'train_result_fold{fold}.csv'))
        
        
        final_results = pd.concat(results_df,0)
        final_results.to_csv(f'final_results.csv')
        for model in train_result_dict:
            # plot scatter
            metrics = metric_func(final_results[model], final_results["Activity"],split=f'{savedir}/train-{model}',plot=True)
            
  
if __name__ == '__main__':
    main()
    print('done.')
