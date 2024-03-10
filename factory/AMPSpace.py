from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import xgboost as xgb
import ipdb


class LstmNet(nn.Module):
    def __init__(self, embedding_dim, hidden_num, num_layer, bidirectional, dropout):
        super(LstmNet, self).__init__()
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_num, num_layer, bidirectional=bidirectional, batch_first=True,
                                  dropout=dropout)  # enbedding_dim=50,hidden_num=32,num_layer=2,bidirectional=True
        self.linear = nn.Sequential(
            nn.Linear(hidden_num * (2 if bidirectional == True else 1), 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1), )
        self.embedding = torch.nn.Embedding(num_embeddings=len(self.get_dict())+1, embedding_dim=embedding_dim,
                                            padding_idx=0)
        
        
        
    def get_dict(self):
      Letter_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
                     'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
      return Letter_dict
        
    def forward(self, sequence):
        
        x = sequence['x']
        x_org = x.detach().clone()
        length = sequence['length'].cpu().int()
        
        x = self.embedding(x.long())
        
        # lengths = torch.tensor([self.length]*x.shape[0])
        # ipdb.set_trace()
        x = pack_padded_sequence(input=x, lengths=length, batch_first=True, enforce_sorted=False)
        output, (h_s, h_c) = self.lstm(x)
        output = pad_packed_sequence(output, batch_first=True)[0]
        out = self.linear(output.mean(dim=1))
        return out, x_org

'''
def XgbClassify():
    model = xgb.XGBClassifier(max_depth=4, n_estimators=600, learning_rate=0.1, use_label_encoder=False,
                              objective="binary:logistic")
    return model


def XgbRank():
    model = xgb.XGBRegressor(max_depth=7, n_estimators=200, learning_rate=0.1, use_label_encoder=False,
                             objective="rank:pairwise")
    return model
'''
