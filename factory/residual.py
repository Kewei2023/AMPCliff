import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Second fully connected layer that outputs our desired output_dim
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Apply the first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply the second fully connected layer
        x = self.fc2(x)
        # Apply sigmoid activation function for binary classification
        # x = torch.sigmoid(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

class ResidualCNN(nn.Module):
    def __init__(self, input_channels, block_channels, kernel_size, stride, padding):
        super(ResidualCNN, self).__init__()
        layers = []
        in_channels = input_channels
        for channels in block_channels:
            layers.append(ResidualBlock(in_channels, channels, kernel_size, stride, padding))
            in_channels = channels
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.network(x)