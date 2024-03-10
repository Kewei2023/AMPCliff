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
        
        return ({k:v[idx] for k,v in self.sequence.items()}, {self.seqName[idx]:self.seqID[idx]}, {k:v[idx] for k,v in self.label_dict.items()})
    
class PSDataset(Dataset):
    '''
    remain to be modified
    '''
    def __init__(self, sequence1: Dict, sequence2: Dict, pairID:List,label_dict: Dict, local_rank: int):
        super(PSDataset).__init__()

        self.sequence1 = sequence1
        self.sequence2 = sequence2
        # self.seqName = seqName
        self.label_dict = label_dict
        self.pairID = pairID

        self.local_rank = local_rank
        # ipdb.set_trace()
    def __len__(self):
        return len(self.pairID)

    def __getitem__(self, idx):
        
        return ({k:v[idx] for k,v in self.sequence1.items()}, 
                {k:v[idx] for k,v in self.sequence2.items()}, 
                {'pairID':self.pairID[idx]}, 
                {k:v[idx] for k,v in self.label_dict.items()}
                ) 


'''
class PreTrainingDataset(Dataset):
    def __init__(self,
                 tokenizer: BertTokenizer,
                 folder: str,
                 logger,
                 max_seq_length,
                 index,
                 data_type: PretrainDataType = PretrainDataType.NUMPY,
                 max_predictions_per_seq: int = 20):
        self.tokenizer = tokenizer
        self.dir_path = folder
        self.max_seq_length = max_seq_length
        self.len = 0
        self.masked_lm_prob = 0.15
        self.max_predictions_per_seq = max_predictions_per_seq
        self.vocab_words = list(tokenizer.vocab.keys())

        path = get_random_partition(self.dir_path, index)

        logger.info(f"Loading Pretraining Data from {path}")
        start = time.time()
        # logger.info(f"Loading Pretraining Data from {path}")
        # if data_type == PretrainDataType.CLEAN_BODY:
        #     self.data = CleanBodyDataCreator.load(path)
        # elif data_type == PretrainDataType.WIKIPEDIA or data_type == PretrainDataType.BOOK_CORPUS:
        #     self.data = WikiNBookCorpusPretrainingDataCreator.load(path)
        if data_type == PretrainDataType.VALIDATION:
            self.data = WikiPretrainingDataCreator.load(path)
        elif data_type == PretrainDataType.NUMPY:
            self.data = NumpyPretrainingDataCreator.load(path)
        self.len = len(self.data)
        logger.info(
            f"Data Loading Completed for Pretraining Data from {path} with {self.len} samples took {time.time()-start:.2f}s."
        )

        self.len = len(self.data)
        logger.info(
            f"Data Loading Completed for Pretraining Data from {path} with {self.len} samples."
        )

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        i = index % self.len

        instance: TokenInstance = self.data.instances[i]
        return self.create_training_instance(instance)

    def create_training_instance(self, instance: TokenInstance):
        tokens_a, tokens_b, is_next = instance.get_values()
        # print(f'is_next label:{is_next}')
        # Create mapper
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)

        tokens.append("[SEP]")
        segment_ids.append(1)

        # Get Masked LM predictions
        tokens, masked_lm_output = self.create_masked_lm_predictions(tokens)

        # Convert to Ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(PAD)
            segment_ids.append(PAD)
            input_mask.append(PAD)
            masked_lm_output.append(-1)
        return ([
            map_to_torch([BatchType.PRETRAIN_BATCH]),
            map_to_torch(input_ids),
            map_to_torch(input_mask),
            map_to_torch(segment_ids),
            map_to_torch([is_next]),
            map_to_torch(masked_lm_output)
        ])

    def create_masked_lm_predictions(self, tokens):
        cand_indexes = []
        for i, token in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            cand_indexes.append(i)

        random.shuffle(cand_indexes)
        output_tokens = list(tokens)

        num_to_predict = min(
            self.max_predictions_per_seq,
            max(1, int(round(len(tokens) * self.masked_lm_prob))))

        masked_lms = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% mask
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% Keep Original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% replace w/ random word
                else:
                    masked_token = self.vocab_words[random.randint(
                        0,
                        len(self.vocab_words) - 1)]

            output_tokens[index] = masked_token
            masked_lms.append(
                MaskedLMInstance(index=index, label=tokens[index]))

        masked_lms = sorted(masked_lms, key=lambda x: x.index)
        masked_lm_output = [-1] * len(output_tokens)
        for p in masked_lms:
            masked_lm_output[p.index] = self.tokenizer.vocab[p.label]

        return (output_tokens, masked_lm_output)
'''