import torch.nn as nn

class AvgPool1d(nn.Module):
    def __init__(self, kernel_size, stride, padding, ceil_mode, count_include_pad):
        super(AvgPool1d, self).__init__()
        self.pool = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)
        
    def forward(self, x):
        # x has shape [batch_size, num_channels, sequence_length]
        x = self.pool(x)
        return x

def averagePooling1dLayer(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    return AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)