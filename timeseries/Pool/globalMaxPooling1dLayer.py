import torch
import torch.nn as nn

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
        
    def forward(self, x):
        # x has shape [batch_size, num_channels, sequence_length]
        x = torch.max(x, dim=-1)[0]
        return x
    
def globalMaxPooling1dLayer():
    return GlobalMaxPool1d()
