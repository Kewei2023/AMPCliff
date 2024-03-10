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
from .dataset import LSDataset,PSDataset
# from.split import random_split_data
from .preprocess_data import clear_generate_data,clear_rank_data,clear_binary_data,clear_regression_data,clear_pretrained_data,clear_ar_data
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

#  use for RankModel
def collate_fn_pair(batch):
    '''
    remain to change
    # sequence1,name2id1, label_dict1,
    # sequence2,name2id2, label_dict2
    '''
    seqs1 = [_[0] for _ in batch]
    seqs2 = [_[1] for _ in batch]
    pairIDs = [_[2] for _ in batch]

    labels= [_[3] for _ in batch]
    

    # seq1
    tmp_dict = defaultdict(list)
    seq_dict1 = {}
    
    for d_ in seqs1:
        for k, v in d_.items():
            tmp_dict[k].append(v)

    for k, v in tmp_dict.items():
        seq_dict1[k] = v

    tmp_dict = defaultdict(list)
    seq_dict2 = {}
    
    for d_ in seqs2:
        for k, v in d_.items():
            tmp_dict[k].append(v)

    for k, v in tmp_dict.items():
        seq_dict2[k] = v

    tmp_dict = defaultdict(list)
    pairids = {}
    
    for d_ in pairIDs:
        for k, v in d_.items():
            tmp_dict[k].append(v)

    for k, v in tmp_dict.items():
        pairids[k] = v

    tmp_dict = defaultdict(list)
    label_dicts = {}
    
    for d_ in labels:
        for k, v in d_.items():
            tmp_dict[k].append(v)

    for k, v in tmp_dict.items():
        label_dicts[k] = v
   
    return seq_dict1,seq_dict2, pairids, label_dicts




def load_raw_seq(dataset_file,tokenizer,cfg):

 
  all_data = pd.read_csv(dataset_file)
#   ipdb.set_trace()
  all_data = all_data.sample(frac=1).reset_index(drop=True)

  if cfg.task.type == "binary":
    sequence, seqName, seqID, label_dict = clear_binary_data(all_data)
  
  if cfg.task.type == "regression":
    
    sequence, seqName, seqID, label_dict = clear_regression_data(all_data,cfg)
  
  if cfg.task.type == "rank":
    sequence1, sequence2, pairID, label_dict = clear_rank_data(all_data,cfg)
    # ipdb.set_trace()
  if cfg.task.type == "generation":
    sequence, seqName, seqID, label_dict = clear_generate_data(all_data,cfg)
  
  if cfg.task.type == "ar":

    all_data['Idx'] = all_data['Idx'].apply(str)
    flipped_df = all_data.copy()
    flipped_df['ID'] = flipped_df['ID'] + '_rev'
    flipped_df['Idx'] = flipped_df['Idx'] + '_rev'
    flipped_df['Sequence'] = flipped_df['Sequence'].apply(lambda s: s[::-1])
    
    all_data = pd.concat([all_data, flipped_df], ignore_index=True)

    all_data = all_data.sample(frac=1).reset_index(drop=True)
    # ipdb.set_trace()
    sequence, seqName, seqID, label_dict = clear_ar_data(all_data,cfg)
  if cfg.task.type == "pretrain":
    sequence, seqName, seqID, label_dict = clear_pretrained_data(all_data,tokenizer,cfg)
    # ipdb.set_trace()
  
  if cfg.task.type == "rank":
      assert len(sequence1) == len(pairID) 
      return sequence1, sequence2, pairID, label_dict
  
  assert len(sequence) == len(seqName) == len(seqID)
  return sequence, seqName, seqID, label_dict





def generate_dataloader(sequence, seqName, seqID, label_dict, ddp, local_rank, batch_size, num_workers, pin_memory, collate_fn):

    for k, v in label_dict.items():
        Logger.info("label_dict | key: {}: shape: {}".format(k, len(v)))


    dataset = LSDataset(sequence, seqName, seqID, label_dict, local_rank)
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

def generate_dataloader_pair(sequence1,sequence2, pairID, label_dict,\
                                ddp, local_rank, batch_size, num_workers, pin_memory, collate_fn):

    for k, v in label_dict.items():
        Logger.info("label_dict | key: {}: shape: {}".format(k, len(v)))

    dataset = PSDataset(sequence1,sequence2, pairID, label_dict, local_rank)
    # ipdb.set_trace()
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
                tokenizer,
                cfg,
                pin_memory=False,
                num_workers=1,
                random_seed=0):
    
    Logger.info(f'start loading ...')

    random.seed(random_seed)



    if cfg.task.type == 'rank':
        sequence1, sequence2, pairID, label_dict  = load_raw_seq(dataset_file,tokenizer,cfg)
        
        def tokenize_function(examples):
            return tokenizer(examples,max_length=cfg.data.max_length, padding="max_length", truncation=True,add_special_tokens=False)
        
        tokenized_sequences1 = list(map(tokenize_function,sequence1))
        tokenized_sequences2 = list(map(tokenize_function,sequence2))
        input_ids_list, attention_mask_list = [], []
        
        for tokenized_sequence in tqdm(tokenized_sequences1):
            input_ids_list.append(tokenized_sequence['input_ids'])
            attention_mask_list.append(tokenized_sequence['attention_mask'])

        tokenized_sequences_dict1 = {'input_ids':input_ids_list, 'attention_mask':attention_mask_list}

        input_ids_list, attention_mask_list = [], []
        
        for tokenized_sequence in tqdm(tokenized_sequences2):
            input_ids_list.append(tokenized_sequence['input_ids'])
            attention_mask_list.append(tokenized_sequence['attention_mask'])

        tokenized_sequences_dict2 = {'input_ids':input_ids_list, 'attention_mask':attention_mask_list}

        dataloader = generate_dataloader_pair(tokenized_sequences_dict1,\
                                          tokenized_sequences_dict2, pairID, label_dict,\
                                          cfg.mode.ddp, local_rank, batch_size, num_workers, pin_memory, collate_fn_pair)
      
    else:
        sequence, seqName, seqID, label_dict = load_raw_seq(dataset_file,tokenizer,cfg)
        

        if cfg.task.type == 'ar':
            def tokenize_function(examples):
                return tokenizer(examples,max_length=cfg.data.max_length, padding="max_length", truncation=True)
            tokenized_sequences = list(map(tokenize_function,sequence))
        
        if cfg.task.type == 'pretrain':
            def tokenize_function(examples):
                return tokenizer(examples,max_length=cfg.data.max_length, padding="max_length", truncation=True,add_special_tokens=False)
            tokenized_sequences = list(map(tokenize_function,sequence))

        if cfg.task.type == 'regression':
            def tokenize_function(examples):
                return tokenizer(examples,max_length=cfg.data.max_length, padding="max_length", truncation=True)
            tokenized_sequences = list(map(tokenize_function,sequence))
        
        if cfg.task.type == 'binary':
            def tokenize_function(examples):
                return tokenizer(examples,max_length=cfg.data.max_length, padding="max_length", truncation=True)
            tokenized_sequences = list(map(tokenize_function,sequence))

        if cfg.task.type == 'generation':
            def tokenize_function(examples):
                return tokenizer(examples,max_length=cfg.data.max_length, padding="max_length", truncation=True)
            tokenized_sequences = list(map(tokenize_function,sequence))

        # ipdb.set_trace()
        input_ids_list, attention_mask_list = [], []
        for tokenized_sequence in tqdm(tokenized_sequences):
            input_ids_list.append(tokenized_sequence['input_ids'])
            attention_mask_list.append(tokenized_sequence['attention_mask'])
            # ipdb.set_trace()
        tokenized_sequences_dict = {'input_ids':input_ids_list, 'attention_mask':attention_mask_list}
        # ipdb.set_trace()
        if cfg.task.type == 'ar':
            label_dict['ar'] = input_ids_list
        # ipdb.set_trace()

        
        dataloader = generate_dataloader(tokenized_sequences_dict, seqName, seqID, label_dict, cfg.mode.ddp, local_rank, batch_size, num_workers, pin_memory, collate_fn)
      
    return dataloader
 
