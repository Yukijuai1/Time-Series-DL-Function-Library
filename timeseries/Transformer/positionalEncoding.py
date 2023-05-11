import numpy as np
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len):
        super(PositionalEncoding, self).__init__()
        self.d = d
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x * np.sqrt(self.d)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

def positionalEncoding(size,max_len=5000):
    return PositionalEncoding(size,max_len)