import torch
import torch.nn as nn
import torch.nn.functional as F
# from .residual import ResidualCNN,FeedForward
import ipdb

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features.mean(1)  # take <s> token (equiv. to mean)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class RegModel_v1(nn.Module):
    '''
    remain modification
    '''
    def __init__(self, pretrain_model,config): # cfg.model.rank.residual
        super(RegModel_v1, self).__init__()
        self.pretrain_model = pretrain_model
        # Assume esm_model outputs features of size N
        # self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        # Regression task layers
        
        
        self.residualcnn_reg =ClassificationHead(config)
        
        # self.regression1 = FeedForward(hidden_dim,hidden_dim,1)
        # self.regression2 = FeedForward(hidden_dim,hidden_dim,1)
        # self.rank = CrossAttentionFeedForward(hidden_dim,hidden_dim,1)
    def forward(self, batch1):
        # Feature extraction from sequences
        features1 = self.pretrain_model(**batch1).last_hidden_state # Assuming last layer output
        regression_output1 = self.residualcnn_reg(features1)
        
        return regression_output1,features1
    





