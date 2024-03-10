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
from AMPCliff.loader.utils_test import make_loader, load_raw_seq #######
from AMPCliff.loader.split import random_split_data,stratified_split_data,fixed_cluster_split_data
from AMPCliff.utils.utils import get_device,fix_random_seed, load_weights, load_model
from AMPCliff.features.feature_fetcher import FeatureFetcher
from AMPCliff.models.breeze import BreezeModel,BreezeTokenizer,BreezeForMaskedLM,BreezeForSequenceClassification, BreezeConfig
from AMPCliff.factory.initializer import ModelInitializer
from AMPCliff.factory.ML import ModelRegressor
from transformers import EsmModel,EsmForSequenceClassification,LlamaForCausalLM, LlamaTokenizer, EsmForMaskedLM
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from AMPCliff.utils.metrics import Metrics
from AMPCliff.visualization.plot import plot_low_dimension
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
       

    fix_random_seed(random_seed, cuda_deterministic=True)
    
    if cfg.logger.log:

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
  
    feature_fetcher = FeatureFetcher(cfg,tokenizer=None)
    
    models = ModelRegressor(random_state=random_seed)
    
    scaler = StandardScaler()
    
    if cfg.data[cfg.task.type].mode == 'fix':
      
      train_sequence, train_seqName, train_seqID, train_label_dict = load_raw_seq(dataset_file=cfg.data[cfg.task.type].fix.all_train_file,
            vocab_dict=None,
            cfg=cfg)
            
      valid_sequence, valid_seqName, valid_seqID, valid_label_dict = load_raw_seq(dataset_file=cfg.data[cfg.task.type].fix.test_file,
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
                  

      df = pd.read_csv(cfg.data[cfg.task.type].fix.test_file)

      pred_df = pd.DataFrame.from_dict(valid_result_dict,orient='index').T
      
      # ipdb.set_trace()
      res_df = pd.concat([df,pred_df], axis=1)

      res_df.to_csv('./test_result.csv')
      # ipdb.set_trace()
      

      # save train result           
      df = pd.read_csv(cfg.data[cfg.task.type].fix.all_train_file)
      
      pred_df = pd.DataFrame.from_dict(train_result_dict,orient='index').T
      
      res_df = pd.concat([df,pred_df], axis=1)

      res_df.to_csv('./train_result.csv')
      
      for model in train_result_dict:
          # plot scatter
          metrics = metric_func(train_result_dict[model], np.array(train_label_dict['regression']),split=f'train-{model}',plot=True)
          metrics = metric_func(valid_result_dict[model], np.array(valid_label_dict['regression']),split=f'test-{model}',plot=True)
      

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
    
        
        all_y_true, all_y_pred = [], []
        results_df = []
        all_latent = []

        result_dir = 'folds'
        os.makedirs(result_dir,exist_ok=True)

        for fold in range(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold):
            
            print("*"*20+f'FOLD {fold}'+"*"*20)
       
            train_sequence, train_seqName, train_seqID, train_label_dict = load_raw_seq(dataset_file=os.path.join(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold_files,f'train_fold_{fold}.csv'),
            vocab_dict=None,
            cfg=cfg)
            
            valid_sequence, valid_seqName, valid_seqID, valid_label_dict = load_raw_seq(dataset_file=os.path.join(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold_files,f'test_fold_{fold}.csv'),
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
                        

            df = pd.read_csv(os.path.join(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold_files,f'test_fold_{fold}.csv'))

            pred_df = pd.DataFrame.from_dict(valid_result_dict,orient='index').T
            
            # ipdb.set_trace()
            res_df = pd.concat([df,pred_df], axis=1)

            res_df.to_csv(f'{result_dir}/test_fold_{fold}.csv')
            # ipdb.set_trace()
            results_df.append(res_df)

            # save train result           
            df = pd.read_csv(os.path.join(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold_files,f'train_fold_{fold}.csv'))
            
            pred_df = pd.DataFrame.from_dict(train_result_dict,orient='index').T
            
            res_df = pd.concat([df,pred_df], axis=1)

            res_df.to_csv(f'{result_dir}/train_fold_{fold}.csv')
            
            all_y_true.append(valid_label_dict)
            all_y_pred.append(valid_result_dict)
            # ipdb.set_trace()
        # ipdb.set_trace()
       
        # save results
        final_results = pd.concat(results_df,0)

        # label_quality_scores = get_label_quality_scores(labels=all_y_true, predictions=all_y_pred)
        # final_results['is_label_issue'] = label_quality_scores
        final_results.to_csv('final_results.csv')
        
        # ipdb.set_trace()
        all_y_true_updated = {key: np.concatenate([np.array(_[key]) for _ in all_y_true]) for key in all_y_true[0]}

        all_y_pred_updated = {key: np.concatenate([_[key] for _ in all_y_pred]) for key in all_y_pred[0]}

        for model in all_y_pred_updated:
          # plot scatter
          metrics = metric_func(all_y_pred_updated[model], all_y_true_updated['regression'],split=f'final-{model}',plot=True)

    
      
if __name__ == '__main__':
    main()
    print('done.')
