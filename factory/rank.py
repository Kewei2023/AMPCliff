import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual import ResidualCNN
from ..models.breeze import BreezeClassificationHead
import ipdb
class RankModel(nn.Module):
    def __init__(self, pretrain_model,config): # cfg.model.rank.residual
        super(RankModel, self).__init__()
        self.pretrain_model = pretrain_model
        
        self.regression = BreezeClassificationHead(config)
        # self.regression2 = BreezeClassificationHead(config)
        self.regression12 = BreezeClassificationHead(config)
        # self.regression = nn.Linear(cfg.block_channels[-1], 1)
        
        # self.residualcnn_rank = ResidualCNN(
        #     input_channels=cfg.input_channels, 
        #     # num_blocks=cfg.num_blocks, 
        #     block_channels=cfg.block_channels, 
        #     kernel_size=cfg.kernel_size, 
        #     stride=cfg.stride, 
        #     padding=cfg.padding
        # )
        
        # self.regression1 = FeedForward(hidden_dim,hidden_dim,1)
        # self.regression2 = FeedForward(hidden_dim,hidden_dim,1)
        # self.rank = CrossAttentionFeedForward(hidden_dim,hidden_dim,1)
    def forward(self, batch1, batch2):
        # Feature extraction from sequences
        features1 = self.pretrain_model(**batch1).last_hidden_state  # Assuming last layer output
        features2 = self.pretrain_model(**batch2).last_hidden_state  # Assuming last layer output

        mqTrans_latent = features1 - features2
        # Rank Loss Task
        # You can further process attn_output as needed for the rank loss task
        # ...
        
        regression_output1 = self.regression(features1)  # Mean over sequence length
        regression_output2 = self.regression(features2)
        regression_output12 = self.regression12(mqTrans_latent)
        
        return regression_output1, regression_output2,regression_output12
