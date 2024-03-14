import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sn
import os
import random
import ipdb
from torch.backends import cudnn


def get_device(cfg):
    device = torch.device(
        "cuda:{}".format(cfg.train.device_ids[0]) if torch.cuda.is_available()
        and len(cfg.train.device_ids) > 0 else "cpu")
    return device


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel,
                           nn.parallel.DistributedDataParallel)


def show_batch(x, y, shape=None):
    """
    input: 
        x(Tensor[num_images, rows, columns]): images tensor
        y(array): labels
        shape(tuple): (rows,col) 
    output:
        grid of smaple images
    """
    if not shape:
        shape = (int(x.shape[0]**0.5), int(x.shape[0]**0.5))

    fig, axs = plt.subplots(nrows=shape[0], ncols=shape[1], figsize=(12, 8))
    index = 0
    for row in axs:
        for ax in row:
            ax.imshow(x[index])
            ax.set_xlabel(y[index], )
            index += 1
    # plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
    fig.tight_layout()
    plt.show()


def save_cm(array, save_name):
    df_cm = pd.DataFrame(array)

    plt.figure(figsize=(10, 10))
    svm = sn.heatmap(df_cm,
                     annot=True,
                     cmap='coolwarm',
                     linecolor='white',
                     linewidths=1)
    plt.savefig(save_name, dpi=400)


def load_weights(model, best_model_path, device):

    best_model_path = best_model_path / "data/model.pth"

    print(best_model_path / "data/model.pth")

    if is_parallel(model):
        model = model.module

    model_dict = model.state_dict()

    best_state_dict = {
        k.replace("module.", ""): v
        for (k, v) in list(
            torch.load(best_model_path,
                       map_location="cpu").state_dict().items())
    }

    model_dict.update(best_state_dict)
    model.load_state_dict(model_dict)

    expected_dict = model.state_dict()
    model.load_state_dict(model_dict, strict=False)
    actual_dict = model.state_dict()
    if set(expected_dict.keys()).issubset(actual_dict.keys()):
        model.to(device)
        return model
    else:
        raise "Keys does not match"

def load_model(model, best_model_path, device):

    best_model_path = best_model_path + "/data/model.pth"

    print(best_model_path + "/data/model.pth")

    if is_parallel(model):
        model = model.module

    model_dict = model.state_dict()

    best_state_dict = {
        k.replace("module.", ""): v
        for (k, v) in list(
            torch.load(best_model_path,
                       map_location="cpu").state_dict().items())
    }

    model_dict.update(best_state_dict)
    # model.load_state_dict(model_dict)

    expected_dict = model.state_dict()
    model.load_state_dict(model_dict, strict=False)
    actual_dict = model.state_dict()
    if set(expected_dict.keys()).issubset(actual_dict.keys()):
        model.to(device)
        return model
    else:
        raise "Keys does not match"
    
def fix_random_seed(random_seed, cuda_deterministic=True):
    # fix random seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True



def seq2pair(testset):
  sequence_pairs = list(combinations(testset.iterrows(), 2))
  
  return 