from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from Bio import SeqIO
import pandas as pd
import numpy as np
import ipdb
import torch
import time
import datetime
import random
import os
from tqdm import tqdm
from functools import partial
from .dataset import LSDataset, ACDataset
from .preprocess_data import clear_regression_data
from ..utils.std_logger import Logger
import re



def collate_fn(batch):
    '''
    sequence,name2id, label_dict
    '''
    seqs = [_[0] for _ in batch]
    name2id_list = [_[1] for _ in batch]
    label_dicts_list = [_[2] for _ in batch]

    tmp_dict = defaultdict(list)
    name2id = {}
    
    for d_ in name2id_list:
        for k, v in d_.items():
            tmp_dict[k].append(v)

    for k, v in tmp_dict.items():
        name2id[k] = v

    tmp_dict = defaultdict(list)
    label_dict = {}
    
    for d_ in label_dicts_list:
        for k, v in d_.items():
            tmp_dict[k].append(v)

    for k, v in tmp_dict.items():
        label_dict[k] = v
    
    tmp_dict = defaultdict(list)
    seq_dict = {}
    
    for d_ in seqs:
        for k, v in d_.items():
            tmp_dict[k].append(v)

    for k, v in tmp_dict.items():
        seq_dict[k] = v

    return seq_dict, name2id, label_dict


def load_raw_seq(dataset_file,vocab_dict,cfg):

 
  all_data = pd.read_csv(dataset_file)
#   ipdb.set_trace()
  all_data = all_data.sample(frac=1).reset_index(drop=True)

  
  if cfg.task.type == "regression":
    
    sequence, seqName, seqID, label_dict = clear_regression_data(all_data,cfg)
  
  
  assert len(sequence) == len(seqName) == len(seqID)
  return sequence, seqName, seqID, label_dict



def generate_dataloader(sequence, seqName, seqID, label_dict, ddp, local_rank, batch_size, num_workers, pin_memory, collate_fn, cfg=None):

    for k, v in label_dict.items():
        Logger.info("label_dict | key: {}: shape: {}".format(k, len(v)))


    dataset = LSDataset(sequence, seqName, seqID, label_dict, local_rank)
    if ddp:
        # ipdb.set_trace()
        sampler = DistributedSampler(dataset, num_replicas = torch.distributed.get_world_size(), rank = torch.distributed.get_rank())
        shuffle = None
    else:
        sampler = None
        shuffle = _dataloader_shuffle(cfg) if cfg is not None else True
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=shuffle,
                            drop_last=False,
                            collate_fn=collate_fn,
                            sampler=sampler)
    return dataloader


def _dataloader_shuffle(cfg) -> bool:
    analysis = cfg.get("analysis", None) if hasattr(cfg, "get") else None
    if analysis is not None:
        manifest = str(getattr(analysis, "peptide_manifest", "") or "").strip()
        if manifest:
            return bool(getattr(analysis, "shuffle_dataloader", False))
    return True


def generate_pair_dataloader(sequence, seqName, label_dict, ddp, local_rank, batch_size, num_workers, pin_memory, collate_fn):

    for k, v in label_dict.items():
        Logger.info("label_dict | key: {}: shape: {}".format(k, len(v)))


    dataset = ACDataset(sequence, seqName, label_dict, local_rank)
    if ddp:
        # ipdb.set_trace()
        sampler = DistributedSampler(dataset, num_replicas = torch.distributed.get_world_size(), rank = torch.distributed.get_rank())
        shuffle = None
    else:
        sampler = None
        shuffle = True
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=shuffle,
                            drop_last=False,
                            collate_fn=collate_fn,
                            sampler=sampler)
    return dataloader

def make_loader(local_rank,
                dataset_file,
                batch_size,
                vocab_dict,
                cfg,
                pin_memory=False,
                num_workers=1,
                random_seed=0):
    
    Logger.info(f'start loading ...')

    random.seed(random_seed)

    if cfg.task.type == 'regression':
        sequence, seqName, seqID, label_dict = load_raw_seq(dataset_file,vocab_dict,cfg)
   
        sequences = {'peptide':sequence}
        
        dataloader = generate_dataloader(sequences, seqName, seqID, label_dict, cfg.mode.ddp, local_rank, batch_size, num_workers, pin_memory, collate_fn, cfg=cfg)
    
    
    return dataloader
