#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /root/CAMP/model/CAMP.py
# Project: /home/richard/projects/syncorepeppi/model
# Created Date: Saturday, July 30th 2022, 7:53:25 pm
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

import math

import torch
import torch.nn as nn
from .input import FeatureProjection
from .interaction import SelfInteractAttention
from .downsampling import AminoAcid2Seq
from .head import AffinityHead, ContactMapTask,BinaryHead

class SynCorePepPIv2(nn.Module):

    def __init__(self,
                 feature_config,
                 arch_config,
                 task_config):
        
        super(SynCorePepPIv2, self).__init__()

        self.feature_config = feature_config
        self.task_config = task_config
     
        self.feature_proj = FeatureProjection(feature_config, arch_config)
        self.interaction = SelfInteractAttention(feature_config, arch_config)    
        self.aa2seq = AminoAcid2Seq(feature_config, arch_config)

        if task_config.type == "multitask":
            self.aff_head =  AffinityHead(feature_config, arch_config)
            self.contact_head = ContactMapTask(feature_config, arch_config)
            
        elif task_config.type == "affinity":
            self.aff_head =  AffinityHead(feature_config, arch_config)
            
        elif task_config.type == "contact_map":
            self.contact_head = ContactMapTask(feature_config, arch_config)

        elif task_config.type == "binary":
            self.binary_head = BinaryHead(feature_config,arch_config)
    
    def forward(self,  feature_dict):
        res = {}
        # input 
        prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features, prot_seq_num_features, pep_seq_num_features = self.feature_proj(feature_dict)
        
        # interaction
        prot_aa_cat_features, pep_aa_cat_features = self.interaction(prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features)

        if self.task_config.type == "contact_map" or self.task_config.type == "multitask":
            contact_res = self.contact_head(prot_aa_cat_features, pep_aa_cat_features)
            res.update(contact_res)
            
        # downsampling for amino acide features
        prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features = self.aa2seq(prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features)

        if self.task_config.type == "affinity" or self.task_config.type == "multitask":
            aff_res = self.aff_head(prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features, prot_seq_num_features, pep_seq_num_features)
            res.update(aff_res)

        if self.task_config.type == "binary":
            cla_res = self.binary_head(prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features, prot_seq_num_features, pep_seq_num_features)
            res.update(cla_res)
            
        return res

    @property
    def device(self):
        return next(self.parameters()).device