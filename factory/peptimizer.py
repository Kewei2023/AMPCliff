import torch
import torch.nn as nn
import torch.nn.functional as F

class Regressor(nn.Module):
    def __init__(self, n_filters, kernel_size, dropout, input_shape):
        super(Regressor, self).__init__()
        
        # 定义网络层
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=n_filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size)
        self.conv3 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=n_filters * self._calculate_flatten_shape(input_shape, kernel_size), out_features=n_filters)
        self.dense2 = nn.Linear(in_features=n_filters, out_features=1)
        
    def _calculate_flatten_shape(self, input_shape, kernel_size):
        # 计算卷积层输出后的尺寸
        seq_length = input_shape[0]
        # 仅考虑卷积层和池化层的影响，不考虑padding和stride的默认值（1）
        for _ in range(3):  # 三层卷积层
            seq_length = seq_length - (kernel_size - 1)
        return seq_length

    def forward(self, x):
        x = x['x'].float()
        x_org = x.detach().clone()
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = self.flatten(x)
        x = F.softplus(self.dense1(x))
        x = self.dense2(x)
        return x,x_org