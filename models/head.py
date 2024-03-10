#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/syncorepeppi/model/head.py
# Project: /home/richard/projects/syncorepeppi/model
# Created Date: Thursday, September 15th 2022, 11:20:25 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Tue Dec 13 2022
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
from .helpers import get_head_input_size

class AffinityHead(nn.Module):

    def __init__(self, feature_config, arch_config):
        super(AffinityHead, self).__init__()

        head_input_size = get_head_input_size(arch_config, feature_config)
                    
        self.head = nn.Sequential(nn.Linear(head_input_size, arch_config["mlp_hidden"]),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(arch_config["dropout"]),
                                    nn.Linear(arch_config["mlp_hidden"], 1), nn.Sigmoid())

    
    def forward(self, prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features, prot_seq_num_features, pep_seq_num_features):
        
        all_features = []
        for i, _ in enumerate([prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features, prot_seq_num_features, pep_seq_num_features]):
            if isinstance(_, torch.Tensor):
                # print(i, _.shape)
                all_features.append(_)

        pep_prot_feature = torch.cat(all_features,
                                     dim=-1)

        affinity_score = self.head(pep_prot_feature)


        return {"affinity": affinity_score}



class ContactMapTask(nn.Module):

    def __init__(self, feature_config, arch_config):
        super(ContactMapTask, self).__init__()

        prot_input_size = arch_config["prot_embed_size"] * len(feature_config.protein.amino_acid.categorical) + arch_config["hidden_size"]
        pep_input_size = arch_config["pep_embed_size"] * len(feature_config.peptide.amino_acid.categorical) + arch_config["hidden_size"]
        
        self.w_protein = nn.Sequential(
            nn.Linear(prot_input_size, prot_input_size),
            nn.LeakyReLU(0.1)
        )
        self.w_peptide = nn.Sequential(
            nn.Linear(pep_input_size, pep_input_size),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, prot_aa_cat_features, pep_aa_cat_features):

        prot_tensor = self.w_protein(prot_aa_cat_features)
        prot_tensor = prot_tensor.permute((0,2,1))
    
        pep_tensor = self.w_peptide(pep_aa_cat_features)
        
        contact_map = torch.bmm(pep_tensor, prot_tensor)
        contact_map = torch.sigmoid(contact_map)
        
        return {"contact_map": contact_map}


class MultiClassHead(nn.Module):

    def __init__(self, feature_config, arch_config, classes):
        super(MultiClassHead, self).__init__()
        
        self.classes = classes
        head_input_size = get_head_input_size(arch_config, feature_config)
        
        self.head = nn.Sequential(nn.Linear(head_input_size, arch_config["mlp_hidden"]),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(arch_config["dropout"]),
                                    nn.Linear(arch_config["mlp_hidden"], classes))

    
    def forward(self, prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features, prot_seq_num_features, pep_seq_num_features):

        all_features = []
        for i, _ in enumerate([prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features, prot_seq_num_features, pep_seq_num_features]):
            if isinstance(_, torch.Tensor):
                # print(i, _.shape)
                all_features.append(_)

        pep_prot_feature = torch.cat(all_features, dim=-1)
        
        # print(pep_prot_feature.shape)
        return self.head(pep_prot_feature)


class BinaryHead(nn.Module):

    def __init__(self, feature_config, arch_config):
        super(BinaryHead, self).__init__()

        head_input_size = head_input_size = get_head_input_size(arch_config, feature_config)
        
        self.head = nn.Sequential(nn.Linear(head_input_size, arch_config["mlp_hidden"]),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(arch_config["dropout"]),
                                    nn.Linear(arch_config["mlp_hidden"], 1),
                                    )

    
    def forward(self, prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features, prot_seq_num_features, pep_seq_num_features):

        all_features = []
        for i, _ in enumerate([prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features, prot_seq_num_features, pep_seq_num_features]):
            if isinstance(_, torch.Tensor):
                # print(i, _.shape)
                all_features.append(_)

        pep_prot_feature = torch.cat(all_features,
                                     dim=-1)

        classification_score = self.head(pep_prot_feature)


        return {"binary": classification_score}

