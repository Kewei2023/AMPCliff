# maintained by kewei li
import torch
import torch.nn as nn


ALPHABET = 'BCDSQKIPTFNGHLRWAVEYM-'


class CNNRegressor(nn.Module):
    def __init__(self, max_length):
        super(CNNRegressor, self).__init__()
        
        
        # self.padding = nn.ZeroPad2d((0, 0, 5, 5))  
        self.conv1 = nn.Conv1d(len(ALPHABET), 64, kernel_size=5)
        self.pool1 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.dropout2 = nn.Dropout(0.5)
        self.dense1 = nn.Linear(256, 100)
        self.dense2 = nn.Linear(100, 100)
        self.dense3 = nn.Linear(100, 20)
        self.output = nn.Linear(20, 1)
    
    
      
      
    def forward(self, x):
        x = x['x'].float()
        x_org = x.detach().clone()
        x = x.permute(0, 2, 1)
        
        
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout2(x)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = self.output(x)
        return x,x_org

class RNNRegressor(nn.Module):
    def __init__(self):
        super(RNNRegressor, self).__init__()
        self.lstm = nn.LSTM(len(ALPHABET), 500, batch_first=True)
        self.dense1 = nn.Linear(500, 100)
        self.output = nn.Linear(100, 1)

    def forward(self, x):
        x = x['x'].float()
        x_org = x.detach().clone()
        x, _ = self.lstm(x)
        x = x[:, -1, :] 
        x = torch.relu(self.dense1(x))
        x = self.output(x)
        return x,x_org
