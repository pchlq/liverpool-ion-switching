import torch
import torch.nn as nn


class Attention(nn.Module):
    #from https://www.kaggle.com/hanjoonchoe/wavenet-lstm-pytorch-ignite-ver
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


       
class Wave_Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(Wave_Block,self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        
        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2**i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=3, padding=dilation_rate, dilation=dilation_rate))
            self.gate_convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=3, padding=dilation_rate, dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=1))
            
    def forward(self,x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x))*torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i+1](x)
            #x += res
            res = torch.add(res, x)
        return res


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 128
        self.LSTM1 = nn.GRU(input_size=19, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        self.LSTM = nn.GRU(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        #self.attention = Attention(input_size,4000)
        #self.rnn = nn.RNN(input_size, 64, 2, batch_first=True, nonlinearity='relu')
       
    
        self.wave_block1 = Wave_Block(128,16,12)
        self.bn1 = nn.BatchNorm1d(16)
        self.wave_block2 = Wave_Block(16,32,8)
        self.bn2 = nn.BatchNorm1d(32)
        self.wave_block3 = Wave_Block(32,64,4)
        self.bn3 = nn.BatchNorm1d(64)
        self.wave_block4 = Wave_Block(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, 11)
        self.dropout = nn.Dropout(0.1)
        
            
    def forward(self,x):
        x,_ = self.LSTM1(x)
        x = x.permute(0, 2, 1)
        # x = self.bn4(x)
        x = self.wave_block1(x)
        x = self.bn1(x)
        x = self.wave_block2(x)
        x = self.bn2(x)
        x = self.wave_block3(x)
        x = self.bn3(x)
        
        #x,_ = self.LSTM(x)
        x = self.wave_block4(x)
        x = self.bn4(x)
        # x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x,_ = self.LSTM(x)
        # x = self.dropout(x)
        #x = self.conv1(x)
        #print(x.shape)
        #x = self.rnn(x)
        #x = self.attention(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x