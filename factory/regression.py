import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual import ResidualCNN,FeedForward
from ..models.breeze import BreezeClassificationHead
import ipdb

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
        
        
        self.residualcnn_reg =BreezeClassificationHead(config)
        
        # self.regression1 = FeedForward(hidden_dim,hidden_dim,1)
        # self.regression2 = FeedForward(hidden_dim,hidden_dim,1)
        # self.rank = CrossAttentionFeedForward(hidden_dim,hidden_dim,1)
    def forward(self, batch1):
        # Feature extraction from sequences
        features1 = self.pretrain_model(**batch1).last_hidden_state # Assuming last layer output
        # ipdb.set_trace()
        regression_output1 = self.residualcnn_reg(features1)
        
        return regression_output1,features1
    

class RegModel_v2(nn.Module):
    def __init__(self, pretrain_model,cfg): # cfg.model.rank.residual
        super(RegModel_v2, self).__init__()
        self.pretrain_model = pretrain_model
        # Assume esm_model outputs features of size N
        # self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        # Regression task layers
        
        
        self.residualcnn_reg =ResidualCNN(
            input_channels=cfg.input_channels, 
            # num_blocks=cfg.num_blocks, 
            block_channels=cfg.block_channels, 
            kernel_size=cfg.kernel_size, 
            stride=cfg.stride, 
            padding=cfg.padding
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.regression = nn.Linear(cfg.block_channels[-1], 1)
        
        
        # self.regression1 = FeedForward(hidden_dim,hidden_dim,1)
        # self.regression2 = FeedForward(hidden_dim,hidden_dim,1)
        # self.rank = CrossAttentionFeedForward(hidden_dim,hidden_dim,1)
    def forward(self, batch1):
        # Feature extraction from sequences
        features1 = self.pretrain_model(**batch1).last_hidden_state  # Assuming last layer output
        
        # extract high resolution features
        res_reg_latent1 = self.residualcnn_reg(features1)
        
        res_reg_latent1 = self.global_avg_pool(res_reg_latent1)
       
        res_reg_latent1 = res_reg_latent1.view(res_reg_latent1.size(0), -1)
        
        # Regression Task
        regression_output1 = self.regression(res_reg_latent1)  # Mean over sequence length
        # regression_output2 = self.regression(res_reg_latent2)
        
        return regression_output1,res_reg_latent1,features1



