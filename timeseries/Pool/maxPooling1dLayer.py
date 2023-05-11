import torch.nn as nn

class MaxPool1d(nn.Module):
    def __init__(self, kernel_size, stride, padding, dilation, ceil_mode):
        super(MaxPool1d, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size, stride, padding, dilation, ceil_mode)
        
    def forward(self, x):
        # x has shape [batch_size, num_channels, sequence_length]
        x = self.pool(x)
        return x

def maxPooling1dLayer(kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    return MaxPool1d(kernel_size, stride, padding, dilation, ceil_mode)