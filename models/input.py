#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/syncorepeppi/model/feature.py
# Project: /home/richard/projects/syncorepeppi/model
# Created Date: Thursday, September 15th 2022, 11:16:04 am
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
from .helpers import PositionalEncoding
import random

class MaskInput(object):
    def __init__(self,cfg):

        self.cfg = cfg
    def read_data(self):

    def mask_protein_sequence(sequence, mask_ratio):
        masked_sequence = list(sequence)
        seq_len = len(sequence)
        num_masks = int(seq_len * mask_ratio)
        mask_indices = random.sample(range(seq_len), num_masks)
        
        for idx in mask_indices:
            masked_sequence[idx] = '<mask>'
        
        return ''.join(masked_sequence)
class FeatureProjection(nn.Module):

    def __init__(self, feature_config, arch_config):
        super(FeatureProjection, self).__init__()

        self.feature_config, self.arch_config = feature_config, arch_config
        
        self._proj_modules = nn.ModuleDict()
        self.feature_dim_dict = feature_config.dimensions 


        ## Peptide 
        # amino_acid

        for fname in feature_config.peptide.amino_acid.embedding:
            self._proj_modules["{}_{}".format("pep", fname)] = nn.Sequential(
            nn.Linear(self.feature_dim_dict[fname], arch_config["hidden_size"]),
        )
        
        for fname in feature_config.peptide.amino_acid.numerical:
            self._proj_modules["{}_{}".format("pep", fname)] = nn.Sequential(
                nn.Linear(self.feature_dim_dict[fname],  arch_config["pep_num_proj_size"]),
                nn.ReLU()
        )

        for fname in feature_config.peptide.amino_acid.categorical:
            self._proj_modules["{}_{}".format("pep", fname)] = nn.Sequential(nn.Embedding(self.feature_dim_dict[fname], arch_config["pep_embed_size"]), 
                                                                                           PositionalEncoding(d_model = arch_config["pep_embed_size"]))


        ### Protein

        for fname in feature_config.protein.amino_acid.embedding:
            self._proj_modules["{}_{}".format("prot", fname)] = nn.Sequential(
                nn.Linear(self.feature_dim_dict[fname], arch_config["hidden_size"]),
            ) 


        for fname in feature_config.protein.amino_acid.numerical:
            self._proj_modules["{}_{}".format("prot", fname)] = nn.Sequential(
                nn.Linear(self.feature_dim_dict[fname], arch_config["prot_num_proj_size"]),
                nn.ReLU()
            )


        for fname in feature_config.protein.amino_acid.categorical:
            self._proj_modules["{}_{}".format("prot", fname)] = nn.Sequential(nn.Embedding(self.feature_dim_dict[fname], arch_config["prot_embed_size"]),
                                                                              PositionalEncoding(d_model = arch_config["prot_embed_size"]))
        

        ## Sequence Embedding

        ### Peptide

        for fname in feature_config.peptide.sequence.embedding:
            self._proj_modules["{}_{}".format("pep", fname)] = nn.Sequential(
                nn.Linear(self.feature_dim_dict[fname], arch_config["hidden_size"])
            ) 

        for fname in feature_config.peptide.sequence.numerical:
            self._proj_modules["{}_{}".format("pep", fname)] = nn.Sequential(
                nn.Linear(self.feature_dim_dict[fname], arch_config["pep_num_proj_size"]),
                nn.ReLU()    
            )
            

        for fname in feature_config.peptide.sequence.categorical:
            self._proj_modules["{}_{}".format("pep", fname)] = nn.Sequential(
                            nn.Embedding(self.feature_dim_dict[fname], arch_config["pep_embed_size"])
            )
            
        ### Protein

        for fname in feature_config.protein.sequence.embedding:
            self._proj_modules["{}_{}".format("prot", fname)] = nn.Sequential(
                nn.Linear(self.feature_dim_dict[fname], arch_config["hidden_size"])
            ) 


        for fname in feature_config.protein.sequence.numerical:
            self._proj_modules["{}_{}".format("prot", fname)] = nn.Sequential(
                nn.Linear(self.feature_dim_dict[fname], arch_config["prot_num_proj_size"]),
                nn.ReLU()
            )
            
        for fname in feature_config.protein.sequence.categorical:
            self._proj_modules["{}_{}".format("prot", fname)] = nn.Embedding(self.feature_dim_dict[fname], arch_config["prot_embed_size"])


    def forward(self, feature_dict):

        # -------------------- protein.amino_acid.embedding -------------------- 
        
        prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features, prot_seq_num_features, pep_seq_num_features = [], [], [], [], [], [], [], []

        for fname in self.feature_config.protein.amino_acid.embedding:
            features = self._proj_modules["{}_{}".format("prot", fname)](feature_dict["{}_{}".format("prot", fname)])
            prot_aa_embedding.append(features)
        
        # -------------------- peptide.amino_acid.embedding --------------------
        for fname in self.feature_config.peptide.amino_acid.embedding:
            features = self._proj_modules["{}_{}".format("pep", fname)](feature_dict["{}_{}".format("pep", fname)])
            pep_aa_embedding.append(features)

         # -------------------- protein.amino_acid.categorical  -------------------- 

        for fname in self.feature_config.protein.amino_acid.categorical:
            features = self._proj_modules["{}_{}".format("prot", fname)](feature_dict["{}_{}".format("prot", fname)])
            prot_aa_cat_features.append(features)

        # -------------------- peptide.amino_acid.categorical  --------------------

        for fname in self.feature_config.peptide.amino_acid.categorical:
            features = self._proj_modules["{}_{}".format("pep", fname)](feature_dict["{}_{}".format("pep", fname)])
            pep_aa_cat_features.append(features)

        # -------------------- protein.amino_acid.numerical  --------------------

        for fname in self.feature_config.protein.amino_acid.numerical:
            features = self._proj_modules["{}_{}".format("prot", fname)](feature_dict["{}_{}".format("prot", fname)])
            prot_aa_num_features.append(features)

        # -------------------- peptide.amino_acid.numerical  --------------------
        for fname in self.feature_config.peptide.amino_acid.numerical:
            features = self._proj_modules["{}_{}".format("pep", fname)](feature_dict["{}_{}".format("pep", fname)])
            pep_aa_num_features.append(features)

        # -------------------- peptide.sequence.numerical  --------------------

        for fname in self.feature_config.peptide.sequence.numerical:
            features = self._proj_modules["{}_{}".format("pep", fname)](feature_dict["{}_{}".format("pep", fname)])
            pep_seq_num_features.append(features)

        # -------------------- protein.sequence.numerical  --------------------

        for fname in self.feature_config.protein.sequence.numerical:
            features = self._proj_modules["{}_{}".format("prot", fname)](feature_dict["{}_{}".format("prot", fname)])
            prot_seq_num_features.append(features)

        res = []
        for i, _ in enumerate([prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features, prot_seq_num_features, pep_seq_num_features]):
            if len(_) > 0:
                # print(i)
                # print([a.shape for a in _ ])
                res.append(torch.cat(_, dim = -1))
            else:
                res.append(_)
                
        return res

# class FeatureProjection(nn.Module):

#     def __init__(self, feature_config, arch_config):
#         super(FeatureProjection, self).__init__()

#         self.feature_config, self.arch_config = feature_config, arch_config
        
#         self._proj_modules = nn.ModuleDict()
#         self.feature_dim_dict = feature_config.dimensions 


#         ## Peptide 
#         # amino_acid

#         for fname in feature_config.peptide.amino_acid.embedding:
#             self._proj_modules["{}_{}".format("pep", fname)] = nn.Sequential(
#             nn.Linear(self.feature_dim_dict[fname], arch_config["hidden_size"]),
#         )
        
#         for fname in feature_config.peptide.amino_acid.numerical:
#             self._proj_modules["{}_{}".format("pep", fname)] = nn.Sequential(
#                 nn.Linear(self.feature_dim_dict[fname],  arch_config["pep_num_proj_size"]),
#                 nn.ReLU()
#         )

#         for fname in feature_config.peptide.amino_acid.categorical:
#             self._proj_modules["{}_{}".format("pep", fname)] = nn.Embedding(self.feature_dim_dict[fname], arch_config["pep_embed_size"])


#         ### Protein

#         for fname in feature_config.protein.amino_acid.embedding:
#             self._proj_modules["{}_{}".format("prot", fname)] = nn.Sequential(
#                 nn.Linear(self.feature_dim_dict[fname], arch_config["hidden_size"]),
#             ) 


#         for fname in feature_config.protein.amino_acid.numerical:
#             self._proj_modules["{}_{}".format("prot", fname)] = nn.Sequential(
#                 nn.Linear(self.feature_dim_dict[fname], arch_config["prot_num_proj_size"]),
#                 nn.ReLU()
#             )


#         for fname in feature_config.protein.amino_acid.categorical:
#             self._proj_modules["{}_{}".format("prot", fname)] = nn.Embedding(self.feature_dim_dict[fname], arch_config["prot_embed_size"])
        

#         ## Sequence Embedding

#         ### Peptide

#         for fname in feature_config.peptide.sequence.embedding:
#             self._proj_modules["{}_{}".format("pep", fname)] = nn.Sequential(
#                 nn.Linear(self.feature_dim_dict[fname], arch_config["hidden_size"])
#             ) 

#         for fname in feature_config.peptide.sequence.numerical:
#             self._proj_modules["{}_{}".format("pep", fname)] = nn.Sequential(
#                 nn.Linear(self.feature_dim_dict[fname], arch_config["pep_num_proj_size"]),
#                 nn.ReLU()    
#             )
            

#         for fname in feature_config.peptide.sequence.categorical:
#             self._proj_modules["{}_{}".format("pep", fname)] = nn.Embedding(self.feature_dim_dict[fname], arch_config["pep_embed_size"])
            

#         ### Protein

#         for fname in feature_config.protein.sequence.embedding:
#             self._proj_modules["{}_{}".format("prot", fname)] = nn.Sequential(
#                 nn.Linear(self.feature_dim_dict[fname], arch_config["hidden_size"])
#             ) 


#         for fname in feature_config.protein.sequence.numerical:
#             self._proj_modules["{}_{}".format("prot", fname)] = nn.Sequential(
#                 nn.Linear(self.feature_dim_dict[fname], arch_config["prot_num_proj_size"]),
#                 nn.ReLU()
#             )
            
#         for fname in feature_config.protein.sequence.categorical:
#             self._proj_modules["{}_{}".format("prot", fname)] = nn.Embedding(self.feature_dim_dict[fname], arch_config["prot_embed_size"])


#     def forward(self, feature_dict):

#         # -------------------- protein.amino_acid.embedding -------------------- 
        
#         prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features, prot_seq_num_features, pep_seq_num_features = [], [], [], [], [], [], [], []

#         for fname in self.feature_config.protein.amino_acid.embedding:
#             features = self._proj_modules["{}_{}".format("prot", fname)](feature_dict["{}_{}".format("prot", fname)])
#             prot_aa_embedding.append(features)
        
#         # -------------------- peptide.amino_acid.embedding --------------------
#         for fname in self.feature_config.peptide.amino_acid.embedding:
#             features = self._proj_modules["{}_{}".format("pep", fname)](feature_dict["{}_{}".format("pep", fname)])
#             pep_aa_embedding.append(features)

#          # -------------------- protein.amino_acid.categorical  -------------------- 

#         for fname in self.feature_config.protein.amino_acid.categorical:
#             features = self._proj_modules["{}_{}".format("prot", fname)](feature_dict["{}_{}".format("prot", fname)])
#             prot_aa_cat_features.append(features)

#         # -------------------- peptide.amino_acid.categorical  --------------------

#         for fname in self.feature_config.peptide.amino_acid.categorical:
#             features = self._proj_modules["{}_{}".format("pep", fname)](feature_dict["{}_{}".format("pep", fname)])
#             pep_aa_cat_features.append(features)

#         # -------------------- protein.amino_acid.numerical  --------------------

#         for fname in self.feature_config.protein.amino_acid.numerical:
#             features = self._proj_modules["{}_{}".format("prot", fname)](feature_dict["{}_{}".format("prot", fname)])
#             prot_aa_num_features.append(features)

#         # -------------------- peptide.amino_acid.numerical  --------------------
#         for fname in self.feature_config.peptide.amino_acid.numerical:
#             features = self._proj_modules["{}_{}".format("pep", fname)](feature_dict["{}_{}".format("pep", fname)])
#             pep_aa_num_features.append(features)

#         # -------------------- peptide.sequence.numerical  --------------------

#         for fname in self.feature_config.peptide.sequence.numerical:
#             features = self._proj_modules["{}_{}".format("pep", fname)](feature_dict["{}_{}".format("pep", fname)])
#             pep_seq_num_features.append(features)

#         # -------------------- protein.sequence.numerical  --------------------

#         for fname in self.feature_config.protein.sequence.numerical:
#             features = self._proj_modules["{}_{}".format("prot", fname)](feature_dict["{}_{}".format("prot", fname)])
#             prot_seq_num_features.append(features)



#         res = []
#         for i, _ in enumerate([prot_aa_embedding, pep_aa_embedding, prot_aa_cat_features, pep_aa_cat_features, prot_aa_num_features, pep_aa_num_features, prot_seq_num_features, pep_seq_num_features]):
#             if len(_) > 0:
#                 # print(i)
#                 # print(_)
#                 res.append(torch.cat(_, dim = -1))
#             else:
#                 res.append(_)
                
#         return res

