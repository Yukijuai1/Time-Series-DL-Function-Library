import torch
import torch.nn as nn

class GlobalAvgPool1d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool1d, self).__init__()
        
    def forward(self, x):
        # x has shape [batch_size, num_channels, sequence_length]
        x = torch.mean(x, dim=-1)
        return x

def globalAveragePooling1dLayer():
    return GlobalAvgPool1d()