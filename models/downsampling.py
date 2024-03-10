#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/syncorepeppi/model/ downsampling.py
# Project: /home/richard/projects/syncorepeppi/model
# Created Date: Thursday, September 15th 2022, 11:20:06 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Mon Nov 21 2022
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

import torch
import torch.nn as nn
from .helpers import Permute, Squeeze


class GlobalMaxPool(nn.Module):
    
    def __init__(self, feature_config, arch_config):
        super(GlobalMaxPool, self).__init__()
        
        self.prot_global_pool = nn.Sequential(
            Permute(), nn.AdaptiveMaxPool1d(output_size=1),
            Squeeze(dim=-1))
        
        self.pep_global_pool = nn.Sequential(
            Permute(), nn.AdaptiveMaxPool1d(output_size=1),
            Squeeze(dim=-1))

    def forward(self, prot_aa_embedding, pep_aa_embedding):

        if isinstance(prot_aa_embedding, torch.Tensor):
            prot_aa_embedding = self.prot_global_pool(prot_aa_embedding)
        if isinstance(pep_aa_embedding, torch.Tensor):
            pep_aa_embedding = self.pep_global_pool(pep_aa_embedding)

        return prot_aa_embedding, pep_aa_embedding

        
class PepProtCNN(nn.Module):

    def __init__(self, feature_config, arch_config):
        super(PepProtCNN, self).__init__()
        
        self.prot_input_size = arch_config["prot_embed_size"] * len(feature_config.protein.amino_acid.categorical) + arch_config["hidden_size"]
        self.pep_input_size = arch_config["pep_embed_size"] * len(feature_config.peptide.amino_acid.categorical) + arch_config["hidden_size"]

        if self.prot_input_size > 0:
            self.prot_cnn_extractor = nn.Sequential(
                Permute(), nn.Conv1d(self.pep_input_size, self.pep_input_size * arch_config["channel_expand_factor"], kernel_size=arch_config["prot_cnn_kernel"]),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.pep_input_size* arch_config["channel_expand_factor"], self.pep_input_size* arch_config["channel_expand_factor"], kernel_size=arch_config["prot_cnn_kernel"]),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.pep_input_size* arch_config["channel_expand_factor"], self.pep_input_size, kernel_size=arch_config["prot_cnn_kernel"]),
                nn.ReLU(inplace=True), nn.AdaptiveMaxPool1d(output_size=1),
                Squeeze(dim=-1))

        if self.pep_input_size > 0:
            self.pep_cnn_extractor = nn.Sequential(
                Permute(), nn.Conv1d(self.pep_input_size, self.pep_input_size * arch_config["channel_expand_factor"], kernel_size = arch_config["pep_cnn_kernel"]),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.pep_input_size* arch_config["channel_expand_factor"], self.pep_input_size * arch_config["channel_expand_factor"], kernel_size = arch_config["pep_cnn_kernel"]),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.pep_input_size* arch_config["channel_expand_factor"], self.pep_input_size, kernel_size = arch_config["pep_cnn_kernel"]),
                nn.ReLU(inplace=True), nn.AdaptiveMaxPool1d(output_size=1),
                Squeeze(dim=-1))

    def forward(self, prot_aa_cat_features, pep_aa_cat_features):

        if isinstance(prot_aa_cat_features, torch.Tensor):
            prot_aa_cat_features = self.prot_cnn_extractor(prot_aa_cat_features)

        if isinstance(pep_aa_cat_features, torch.Tensor):
            pep_aa_cat_features = self.pep_cnn_extractor(pep_aa_cat_features)

        return prot_aa_cat_features, pep_aa_cat_features




class AminoAcid2Seq(nn.Module):
    
    def __init__(self, feature_config, arch_config):
        super(AminoAcid2Seq, self).__init__()

        self.cnn = PepProtCNN(feature_config, arch_config)
        self.pool = GlobalMaxPool(feature_config, arch_config)

    def forward(self, prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features):

        prot_aa_embedding, pep_aa_embedding = self.pool(prot_aa_embedding, pep_aa_embedding)
        prot_aa_num_features, pep_aa_num_features = self.pool(prot_aa_num_features, pep_aa_num_features)
        prot_aa_cat_features, pep_aa_cat_features = self.cnn(prot_aa_cat_features, pep_aa_cat_features)
        
        return prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features