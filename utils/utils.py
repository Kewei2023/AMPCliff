#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /root/CAMP/utils/utils.py
# Project: /home/richard/projects/syncorepeppi/utils
# Created Date: Saturday, July 30th 2022, 3:49:43 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Sun Dec 04 2022
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2022 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2022 Ruochi Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
###

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

