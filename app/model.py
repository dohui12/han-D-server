import torch
import torch.nn as nn
from .config import param

class KPNet(nn.Module):
    def __init__(self, time_steps, feat_dim, n_classes, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=feat_dim, hidden_size=128, batch_first=True)
        self.bn1   = nn.BatchNorm1d(128)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.fc1   = nn.Linear(128, 64)
        self.drop  = nn.Dropout(dropout)
        self.out   = nn.Linear(64, n_classes)

    def forward(self, x):
        o,_ = self.lstm1(x)        # (B,T,128)
        o = o[:, -1, :]            # (B,128)
        o = self.bn1(o)
        o = torch.relu(o)
        o,_ = self.lstm2(o.unsqueeze(1))  # (B,1,128)
        o = o[:, -1, :]
        o = torch.relu(self.fc1(o))
        o = self.drop(o)
        return self.out(o)

# 모델 로드
def load_model():
    model = KPNet(param['time_steps'], param['feat_dim'], param['n_classes'])
    model.load_state_dict(torch.load(param['model_path'], map_location=param['device']))
    model.to(param['device']).eval()
    return model
