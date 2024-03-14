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



    
