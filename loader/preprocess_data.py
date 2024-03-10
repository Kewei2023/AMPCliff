import os
import pandas as pd
import numpy as np
import ipdb
import random
import collections


def get_vocab_words(cfg):
    my_file = open(os.path.join(cfg.model.dir,'vocab.txt'), "r")
    
    # reading the file
    data = my_file.read()
    
    # replacing end splitting the text 
    # when newline ('\n') is seen.
    vocab_words = data.split("\n")
    print(vocab_words)
    my_file.close()
    return vocab_words

def set_noise_label(activity,noise_level):
        
        original_values = np.power(10,-activity) * 1e6
        var = (original_values.max()-original_values.min()) * noise_level /10

        add_noise = np.zeros(len(activity))

        random.seed(19970813)
        for idx,sample in enumerate(original_values):

            while True:
                noise = random.gauss(0,var**0.5)
                if sample + noise > 0:
                    add_noise[idx] = sample + noise
                    break

        noised_activity = list(-np.log10(add_noise*1e-6))

        return noised_activity

def clear_generate_data(all_data,cfg):
    
    all_data.drop_duplicates(subset = ["Sequence"], inplace=True)
    if cfg.other.debug:
        all_data = all_data.iloc[:8,:]
    sequence = all_data["Sequence"].values.tolist()
    seqName = all_data["ID"].values.tolist()
    # labels =  all_data["Activity"].values.tolist()
    seqID = all_data["Idx"].values.tolist()
    
    label_dict = {}
    
    return sequence, seqName, seqID, label_dict


def clear_binary_data(all_data: pd.DataFrame):
    
    all_data.drop_duplicates(subset = ["Sequence"], inplace=True)

    sequence = all_data["Sequence"].values.tolist()
    seqName = all_data["ID"].values.tolist()
    labels =  all_data["label"].values.tolist()
    seqID = all_data["Idx"].values.tolist()
    
    label_dict = {"binary": labels}
    
    return sequence, seqName, seqID, label_dict

def clear_regression_data(all_data,cfg):
    
    all_data.drop_duplicates(subset = ["Sequence"], inplace=True)
    if cfg.other.debug:
        all_data = all_data.iloc[:4,:]
    sequence = all_data["Sequence"].values.tolist()
    seqName = all_data["ID"].values.tolist()
    activities = all_data["Activity"].values.tolist()
    
    noised_activities = set_noise_label(np.array(activities),cfg.data.regression.noise_level)
    
    seqID = all_data["Idx"].values.tolist()
    # ipdb.set_trace()
    label_dict = {"noised_regression": noised_activities,"regression": activities}
    
    return sequence, seqName, seqID, label_dict

def clear_rank_data(all_data,cfg):
    
    all_data.drop_duplicates(subset = ["seq1",'seq2'], inplace=True)
    if cfg.other.debug:
        all_data = all_data.iloc[:10,:]
    seq1 = all_data["seq1"].values.tolist()
    seq2 = all_data["seq2"].values.tolist()
    pairID = all_data["Idx"].values.tolist()
    # labels = all_data["isnot AC"].values.tolist()
    activity1 =  all_data["mic1"].values.tolist()
    activity2 =  all_data["mic2"].values.tolist()
    activity12 =  all_data["dmic"].values.tolist()

    label_dict = {'regression1':activity1, 'regression2':activity2, 'regression12':activity12}
    
    return seq1,seq2,pairID,label_dict

def clear_ar_data(all_data,cfg):
    
    all_data.drop_duplicates(subset = ["Sequence"], inplace=True)
    if cfg.other.debug:
        all_data = all_data.iloc[:32000,:]
    
    sequences = all_data["Sequence"].values.tolist()

    # sequences = [' '.join(list(seq)) for seq in sequences]

    seqName = all_data["ID"].values.tolist()
    seqID = all_data["Idx"].values.tolist()

    label_dict = {"ar":sequences}
   
    return sequences, seqName, seqID, label_dict

def clear_pretrained_data(all_data,vocab_dict, cfg):
    
    all_data.drop_duplicates(subset = ["Sequence"], inplace=True)
    if cfg.other.debug:
        all_data = all_data.iloc[:300,:]
    sequences = all_data["Sequence"].values.tolist() * cfg.data.pretrain.repeat
    seqName = all_data["ID"].values.tolist() * cfg.data.pretrain.repeat
    seqID = all_data["Idx"].values.tolist() * cfg.data.pretrain.repeat

    masked_seqs = []
    label_dict = {'pretrain':[]}
    # vocab_words = get_vocab_words(cfg)
    for sequence in sequences:
        # try:
        # output_sequence, masked_lm_positions, masked_lm_labels = mask_sequences(sequence,vocab_words, cfg.model.mask_data)
        output_sequence,masked_lm_labels = mask_sequences(sequence,vocab_dict, cfg.model.pretrain.mask_data)
        masked_seqs.append(output_sequence)
        label_dict['pretrain'].append(masked_lm_labels)
        
    return masked_seqs, seqName, seqID, label_dict



def mask_sequences(sequence,vocab_dict,cfg): # cfg.mask_data

    MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])
    tokens = []
    tokens.append("<cls>")

    cutoff = cfg.max_length - 2

    for i, token in enumerate(sequence):
        if i < cutoff :
            tokens.append(token)

    tokens.append("<eos>")

    sequence_length = len(tokens)
    
    cand_indexes = []
    for i, token in enumerate(tokens):
        if token == "<cls>" or token == "<eos>" or token == "<pad>":
            continue
        cand_indexes.append(i)

    random.shuffle(cand_indexes)
    output_tokens = list(tokens)
     
    num_to_predict = min(cfg.max_predictions_per_seq,
                       max(1, int(round(sequence_length * cfg.masked_lm_prob))))
    
    vocab_words = list(vocab_dict.keys())
    # ipdb.set_trace()
    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        rng = random.random()
        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:
            masked_token = "<mask>"

        elif random.random() < 0.5: # 10% of the time, keep original
                masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
            masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token
        # masked_lms += 1
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    # ipdb.set_trace()
    masked_lm_output = [-1] * cfg.max_length
    for p in masked_lms:
        try:
            masked_lm_output[p.index] = vocab_dict[p.label]
        except:
            ipdb.set_trace()

    output_sequence = ''.join(output_tokens)
    # ipdb.set_trace()
    return (output_sequence, masked_lm_output)

    
