import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, kernel_regularizer=None, batch_normalisation=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=not batch_normalisation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=not batch_normalisation)
        self.activation = activation
        self.batch_normalisation = batch_normalisation
        if batch_normalisation:
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_normalisation:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.batch_normalisation:
            x = self.bn2(x)
        x = self.activation(x)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, kernel_regularizer=None, batch_normalisation=False):
        super().__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=1,padding=1)
        
        self.conv = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=not batch_normalisation)
        self.conv_block = ConvBlock(out_channels*2, out_channels, kernel_size, activation, kernel_regularizer, batch_normalisation)
        self.activation = activation
        
        
    def forward(self, x, skip):
        x = self.upconv(x) # 32->16
        x = self.conv(x)
        x =self.activation(x)
        x = torch.cat((skip, x), 1)
        x = self.conv_block(x)
        return x

class UNet(nn.Module):
    def __init__(self, filters=8, kernel_size=5, num_layers=3, dropout=0, graph_layers=None, graph_activation=F.relu, conv_activation=F.relu, kernel_regularizer=None, batch_normalisation=False):
        super().__init__()
        self.num_layers = num_layers - 1 # 0 index layers
        self.contraction_blocks = nn.ModuleList()
        self.expansion_blocks = nn.ModuleList()
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = dropout
        # Contraction path
        for i in range(self.num_layers): # 0,1
            
          in_channels = filters * 2 ** (i-1) if i > 0 else 22 # 22, 8*2^0=8
          out_channels = filters * 2 ** i # 8*2^0=8, 8*2^1=16
          self.contraction_blocks.append(ConvBlock(in_channels, out_channels, kernel_size, conv_activation, kernel_regularizer, batch_normalisation))
            
        # Bottom layer  16->32
        self.bottom_block = ConvBlock(filters * 2 ** (self.num_layers-1), filters * 2 ** (self.num_layers), kernel_size, conv_activation, kernel_regularizer, batch_normalisation)

        # Expansion path
        for i in reversed(range(self.num_layers)): # 3-1=2 -> 1,0
            in_channels = filters * 2 ** (i+1) # 8*2^2 = 32, 8*2^1 = 16
            out_channels = filters * 2 ** (i)  # 8*2^1 = 16, 8*2^0 = 8
            self.expansion_blocks.append(UpConvBlock(in_channels, out_channels, kernel_size, conv_activation, kernel_regularizer, batch_normalisation))

        # Final predictor conv layer
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels, 1)
        

    def apply_max_pool1d_with_same_padding(self,x, kernel_size=2, stride=2):
        
        input_length = x.size(2)  #  [batch_size, channels, length]
        output_length = x.size(2)
        padding_needed = max(0, (output_length - 1) * stride + kernel_size - input_length)
       
        padding_left = padding_needed // 2
        padding_right = padding_needed - padding_left
        
        x_padded = F.pad(x, (padding_left, padding_right), mode='constant', value=0)
        x_pooled = F.max_pool1d(x_padded, kernel_size, stride)
        return x_pooled
    

    def forward(self, x):
        x = x['x'].float()
        x_org = x.detach().clone()
        x = x.permute(0, 2, 1)  
        
        skips = []
        for block in self.contraction_blocks:
            # ipdb.set_trace()
            
            x = block(x)
            
            # ipdb.set_trace()
            
            skips.append(x)
            
            x = self.apply_max_pool1d_with_same_padding(x)
            
            # ipdb.set_trace()
            if self.dropout:
                x = F.dropout(x, p=self.dropout)
        
        # ipdb.set_trace()
        x = self.bottom_block(x)
        
        if self.dropout:
            x = F.dropout(x, p=self.dropout)

        for i, block in enumerate(self.expansion_blocks):
            
            x = block(x, skips[-(i+1)])
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        preds = self.fc(x)
        
        return preds, x_org