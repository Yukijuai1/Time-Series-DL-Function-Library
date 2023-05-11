import torch
import torch.nn as nn

class DropoutLayer(nn.Module):
    def __init__(self, dropout_rate):
        super(DropoutLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        x = self.dropout(x)
        return x
    
def dropoutLayer(dropout_rate=0.5):
    return DropoutLayer(dropout_rate)
