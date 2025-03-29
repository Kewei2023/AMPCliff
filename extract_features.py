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
    

    
    random_seed = cfg.train.random_seed
    device = get_device(cfg)

    if global_rank == 0:
        Logger.info("setting random seed: {}".format(random_seed))

    fix_random_seed(random_seed, cuda_deterministic=True)
    

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

   
        
    # data.regression.mode should be random
    if cfg.data[cfg.task.type].mode != 'random':
        
        Logger.info('change data.regression.mode to random!')
        exit()
    
    if global_rank == 0:
        Logger.info('loading train data...')
    
    train_file_path = cfg.data[cfg.task.type].random.data_file
      
    
    train_dataloader = make_loader(
                                    local_rank=local_rank,
                                    dataset_file=train_file_path,
                                    cfg = cfg,
                                    batch_size=cfg.train.batch_size,
                                    vocab_dict=vocab_dict,
                                    pin_memory=False,
                                    num_workers=cfg.train.num_workers,
                                    random_seed=random_seed
                                )  # change shuffle in the make_loader function to False
        

    dataloaders = {'train': train_dataloader}
    
    
    evaluator = Evaluator(model,dataloaders, metric_func,feature_fetcher, device, cfg)
     
    
    if global_rank == 0:
      Logger.info("evaluating train set......")
    
    evaluate_tr_metrics = evaluator.run('train')
    latent = evaluator.latent
    
    dataset_name = cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].data_file.split('/')[-1].split('.')[0]
    model_name = cfg.model[cfg.task.type].version
    
    save_dir = f'./embedding_features/{dataset_name}/{model_name}'
    os.makedirs(save_dir,exist_ok=True)
    
    np.save(os.path.join(save_dir,'last_layer_features_mean.npy'),latent.numpy())
    
 

    if cfg.mode.ddp:
        cleanup_multinodes()
      
if __name__ == '__main__':
    main()
    print('done.')
