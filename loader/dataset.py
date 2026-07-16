# maintained by kewei li
from torch.utils.data import Dataset
import torch
import math
from typing import List, Dict
import numpy as np 
import random
import ipdb
class LSDataset(Dataset):

    def __init__(self, sequence: Dict, seqName: List,seqID:List, label_dict: Dict,local_rank: int):
        super(LSDataset).__init__()

        self.sequence = sequence
        self.label_dict = label_dict
        self.seqName = seqName
        self.seqID = seqID
        self.local_rank = local_rank
        # ipdb.set_trace()
    def __len__(self):
        return len(self.seqName)

    def __getitem__(self, idx):
        
        return ({k:v[idx] for k,v in self.sequence.items()}, 
                {self.seqName[idx]:self.seqID[idx]}, 
                {k:v[idx] for k,v in self.label_dict.items()})


class ACDataset(Dataset):

    def __init__(self, sequences: Dict,seqName: List, label_dict: Dict,local_rank: int):
        super(ACDataset).__init__()

        self.sequences = sequences
        self.label_dict = label_dict
        self.seqName = seqName
        # self.seqID = seqID
        self.local_rank = local_rank
        # ipdb.set_trace()
    def __len__(self):
        return len(self.seqName)

    def __getitem__(self, idx):
        
        return (
                {k:v[idx] for k,v in self.sequences.items()}, 
                # {k:v[idx] for k,v in self.q_sequence.items()}, 
                {self.seqName[idx]:self.seqName[idx]},
                {k:v[idx] for k,v in self.label_dict.items()})
