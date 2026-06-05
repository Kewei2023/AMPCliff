import os
import sys
import json
import time
import uuid
import hashlib
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
from AMPCliff.utils.path_helper import resolve_path
from AMPCliff.utils.trainer import Trainer ######
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

DEBUG_LOG_PATH = "/data/home/scv6872/run/kwli/AMPCliff/.cursor/debug-26eaa2.log"
DEBUG_SESSION_ID = "26eaa2"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _rng_fingerprint() -> dict:
    try:
        py_state = hashlib.sha256(repr(__import__("random").getstate()).encode("utf-8")).hexdigest()[:16]
    except Exception:
        py_state = "na"
    try:
        np_state = np.random.get_state()
        np_repr = f"{np_state[0]}|{np_state[2]}|{np_state[3]}|{np_state[4]}|{int(np_state[1][0])}"
        np_hash = hashlib.sha256(np_repr.encode("utf-8")).hexdigest()[:16]
    except Exception:
        np_hash = "na"
    try:
        torch_seed = int(torch.initial_seed())
        torch_state = torch.get_rng_state().cpu().numpy()
        torch_hash = hashlib.sha256(torch_state.tobytes()).hexdigest()[:16]
    except Exception:
        torch_seed = -1
        torch_hash = "na"
    return {
        "python_rng_hash": py_state,
        "numpy_rng_hash": np_hash,
        "torch_seed": torch_seed,
        "torch_rng_hash": torch_hash,
    }


########################################################################
# util

@hydra.main(config_path="configs", config_name="downstream.yaml")
def main(cfg: DictConfig):
    
    orig_cwd = hydra.utils.get_original_cwd()
    cfg.orig_cwd = orig_cwd
    run_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    local_rank = 0
    global_rank = 0
    # ipdb.set_trace()
    if not cfg.mode.nni and cfg.logger.log:
        os.environ["NO_PROXY"] = os.environ.get("NO_PROXY","") + ",172.26.114.59,172.26.0.0/16"
        os.environ["no_proxy"] = os.environ["NO_PROXY"]
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
        # #region agent log
        _debug_log(
            hypothesis_id="H1_H2_H3",
            location="downstream_train.py:main:init",
            message="training run initialized",
            data={
                "cwd": os.getcwd(),
                "orig_cwd": str(orig_cwd),
                "pooling": str(cfg.model[cfg.task.type].pooling),
                "argv_pooling_overrides": [a for a in sys.argv if "pooling" in a],
                "frame_weight_version": str(getattr(cfg.model[cfg.task.type], "frame_weight_version", "NA")),
                "model_version": str(cfg.model[cfg.task.type].version),
                "dataset_mode": str(cfg.data[cfg.task.type].mode),
                "dataset": str(cfg.data[cfg.task.type].get("dataset")),
                "seed": int(random_seed),
            },
            run_id=run_id,
        )
        _debug_log(
            hypothesis_id="H7",
            location="downstream_train.py:main:rng_after_seed",
            message="rng fingerprint after fix_random_seed pending",
            data=_rng_fingerprint(),
            run_id=run_id,
        )
        # #endregion

    fix_random_seed(random_seed, cuda_deterministic=True)
    if global_rank == 0:
        # #region agent log
        _debug_log(
            hypothesis_id="H7",
            location="downstream_train.py:main:rng_after_fix_seed",
            message="rng fingerprint after fix_random_seed",
            data=_rng_fingerprint(),
            run_id=run_id,
        )
        # #endregion

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
    
    if global_rank == 0:
        # #region agent log
        _debug_log(
            hypothesis_id="H8",
            location="downstream_train.py:main:rng_before_model_init",
            message="rng fingerprint before model init",
            data=_rng_fingerprint(),
            run_id=run_id,
        )
        # #endregion
    model, tokenizer = ModelInitializer(cfg,device).init()
    if global_rank == 0:
        # #region agent log
        _debug_log(
            hypothesis_id="H8",
            location="downstream_train.py:main:rng_after_model_init",
            message="rng fingerprint after model init",
            data=_rng_fingerprint(),
            run_id=run_id,
        )
        # #endregion
    if cfg.mode.ddp:
        model = DDP(model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True
                    )
                    
    if tokenizer is not None:
      vocab_dict = tokenizer.get_vocab()
    else:
      vocab_dict = None
      
      
    feature_fetcher = FeatureFetcher(cfg,tokenizer)
    
    
    # diff = cfg.data.diff
    # condition = cfg.data[cfg.task.type].condition
    threshold = str(cfg.data.threshold)
    for condition in cfg.data[cfg.task.type].condition:
          
      for diff in cfg.data.diff: #[2,3,4,5]:
        optimizer = AdamW(model.parameters(),
                    lr=cfg.train.learning_rate,
                    betas=(0.9, 0.999),
                    eps=cfg.train.adam_epsilon,
                    weight_decay=cfg.train.weight_decay)
    
        scheduler = lr_scheduler(cfg, optimizer)
        
        # fix data files
        if cfg.data[cfg.task.type].mode == 'fix':
  
          # for condition in cfg.data[cfg.task.type].condition:
            
            # for diff in cfg.data.diff: #[2,3,4,5]:
              
          if global_rank == 0:
              Logger.info('loading train data...')
          
          train_file_path = resolve_path(
                cfg.data[cfg.task.type].fix.train_file,
                diff=diff,
                condition=condition,
                threshold=threshold,
                dataset=cfg.data[cfg.task.type].get('dataset')
            )
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
              # #region agent log
              _debug_log(
                  hypothesis_id="H9",
                  location="downstream_train.py:main:rng_after_train_loader",
                  message="rng fingerprint after train dataloader build",
                  data=_rng_fingerprint(),
                  run_id=run_id,
              )
              # #endregion
          if global_rank == 0:
              Logger.info('loading valid data...')
          
          valid_file_path = resolve_path(
                cfg.data[cfg.task.type].fix.valid_file,
                diff=diff,
                condition=condition,
                threshold=threshold,
                dataset=cfg.data[cfg.task.type].get('dataset')
            )
          
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
          
          test_file_path = resolve_path(
                cfg.data[cfg.task.type].fix.test_file,
                diff=diff,
                condition=condition,
                threshold=threshold,
                dataset=cfg.data[cfg.task.type].get('dataset')
            )
          if global_rank == 0:
              # #region agent log
              _debug_log(
                  hypothesis_id="H4",
                  location="downstream_train.py:main:resolved_paths",
                  message="resolved dataset files",
                  data={
                      "condition": str(condition),
                      "diff": int(diff),
                      "threshold": str(threshold),
                      "train_file": str(train_file_path),
                      "valid_file": str(valid_file_path),
                      "test_file": str(test_file_path),
                  },
                  run_id=run_id,
              )
              # #endregion
          
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
          
          
          if not cfg.model[cfg.task.type].check_point.load:
          
            trainer = Trainer(model, dataloaders, optimizer, scheduler, metric_func,feature_fetcher,
                          device, global_rank, cfg, best_model_dir=f'{cfg.model[cfg.task.type].version}/{condition}/diff{diff}-trd{threshold}')
                
            
            trainer.run()
    
            if global_rank == 0:
                
                Logger.info("finished training......")
                Logger.info("start evaluating......")
                Logger.info("loading best weights from {}......".format(
                    trainer.best_model_path))
                # #region agent log
                _debug_log(
                    hypothesis_id="H5",
                    location="downstream_train.py:main:best_model",
                    message="trainer selected best model",
                    data={
                        "best_model_path": str(trainer.best_model_path),
                        "best_metric": str(trainer.best_metrics[cfg.task[cfg.task.type].default_metric]) if hasattr(trainer, "best_metrics") else "NA",
                    },
                    run_id=run_id,
                )
                # #endregion
                
                model = load_weights(model, trainer.best_model_path, device)
          
          else:
            model = load_model(model, cfg.model[cfg.task.type].check_point.path, device)
          
          evaluator = Evaluator(model,dataloaders, metric_func,feature_fetcher, device, cfg)
          
          
          
          if global_rank == 0:
            Logger.info("evaluating train set......")
          
          evaluate_tr_metrics = evaluator.run('train')
          y_pred,y_true,ids = evaluator.y_pred,evaluator.y_true,evaluator.ids
            
          if cfg.mode.ddp:
            # collect results of each gpu
            gather_list_pred = [torch.zeros_like(y_pred).to(device) for _ in range(world_size)]
            gather_list_true = [torch.zeros_like(y_true).to(device) for _ in range(world_size)]
            gather_list_ids = [torch.zeros_like(ids).to(device) for _ in range(world_size)]
            
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
            metric_func(y_pred,y_true ,split=f'train-{cfg.model[cfg.task.type].version}-{condition}-diff{diff}-trd{threshold}',plot=True)
            # save results
            df = pd.read_csv(train_file_path)
                
            pred_df = pd.DataFrame.from_dict({f'{cfg.model[cfg.task.type].version}-pred':y_pred.squeeze(1).cpu(),
                                              'true':y_true.cpu(),
                                              'Idx':ids})
            res_df = pd.merge(df,pred_df,on='Idx',how='outer').sort_values(by='Idx')
            res_df.to_csv(f'./{cfg.model[cfg.task.type].version}-{condition}-diff{diff}-trd{threshold}-train_result.csv')
          
          
          
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
          
            metric_func(y_pred,y_true ,split=f'valid-{cfg.model[cfg.task.type].version}-{condition}-diff{diff}-trd{threshold}',plot=True)
            # save results
            df = pd.read_csv(valid_file_path)
                
            pred_df = pd.DataFrame.from_dict({f'{cfg.model[cfg.task.type].version}-pred':y_pred.squeeze(1).cpu(),
                                              'true':y_true.cpu(),
                                              'Idx':ids})
            res_df = pd.merge(df,pred_df,on='Idx',how='outer').sort_values(by='Idx')
            res_df.to_csv(f'./{cfg.model[cfg.task.type].version}-{condition}-diff{diff}-trd{threshold}-valid_result.csv')
          
          
          
          
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
            metric_func(y_pred,y_true ,split=f'test-{cfg.model[cfg.task.type].version}-{condition}-diff{diff}-trd{threshold}',plot=True)
            # save results
            df = pd.read_csv(test_file_path)
                
            pred_df = pd.DataFrame.from_dict({f'{cfg.model[cfg.task.type].version}-pred':y_pred.squeeze(1).cpu(),
                                              'true':y_true.cpu(),
                                              'Idx':ids})
            res_df = pd.merge(df,pred_df,on='Idx',how='outer').sort_values(by='Idx')
            res_df.to_csv(f'./{cfg.model[cfg.task.type].version}-{condition}-diff{diff}-trd{threshold}-test_result.csv')
            
            if cfg.data[cfg.task.type].acindex:
              pair_df = seq2pair(res_df,cfg.data.diff)
              pair_df.to_csv(f'./{cfg.model[cfg.task.type].version}-{condition}-diff{diff}-trd{threshold}-test_pair_result.csv')
          
          
          
          if global_rank == 0:
              if not cfg.mode.nni and cfg.logger.log:            
                  for metric_name, metric_v in evaluate_tr_metrics.items():
                      if isinstance(metric_v, (float, np.float64, int)):
                          mlflow.log_metric("train_final/{}".format(metric_name), metric_v, step=1)
                      elif isinstance(metric_v, str):
                          mlflow.log_text(metric_v, "train_final/report.txt")
                  
                  for metric_name, metric_v in evaluate_val_metrics.items():
                      if isinstance(metric_v, (float, np.float64, int)):
                          mlflow.log_metric("valid_final/{}".format(metric_name), metric_v, step=1)
                      elif isinstance(metric_v, str):
                          mlflow.log_text(metric_v, "valid_final/report.txt")
                  
                  for metric_name, metric_v in evaluate_test_metrics.items():
                      if isinstance(metric_v, (float, np.float64, int)):
                          mlflow.log_metric("test_final/{}".format(metric_name), metric_v, step=1)
                      elif isinstance(metric_v, str):
                          mlflow.log_text(metric_v, "test_final/report.txt")
    
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
                
                trainer = Trainer(model, dataloaders, optimizer, scheduler, metric_func,feature_fetcher,
                                device, global_rank, cfg, best_model_dir=f'fold {fold}')
                
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
            
            evaluator = Evaluator(model,dataloaders, metric_func,feature_fetcher, device, cfg)
            
            # save valid result
            evaluator.run('valid')
            
            y_pred,y_true,ids = evaluator.y_pred,evaluator.y_true,evaluator.ids# ,evaluator.names,evaluator.ids,evaluator.latent
            
            
            all_y_pred.append(y_pred)
            all_y_true.append(y_true)
            # all_latent.append(latent)

            df = pd.read_csv(os.path.join(cfg.data[cfg.task.type][cfg.data[cfg.task.type].mode].fold_files,f'test_fold_{fold}.csv'))
            
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
        final_results.to_csv(f'{cfg.model[cfg.task.type].version}_final_results.csv')

        # plot scatter
        metrics = metric_func(all_y_pred, all_y_true,split=f'{cfg.model[cfg.task.type].version}-random-final',plot=True)

        # plot the last hidden_states
        # plot_low_dimension(all_latent, labels=final_results['group'])

        
        if global_rank == 0:
            if not cfg.mode.nni and cfg.logger.log:            
                for metric_name, metric_v in metrics.items():
                    if isinstance(metric_v, (float, np.float64, int, np.int)):
                        mlflow.log_metric("cv_final/{}".format(metric_name), metric_v, step=1)
                    elif isinstance(metric_v, str):
                        mlflow.log_text(metric_v, "cv_final/report.txt")

    if cfg.mode.ddp:
        cleanup_multinodes()
      
if __name__ == '__main__':
    main()
    print('done.')
