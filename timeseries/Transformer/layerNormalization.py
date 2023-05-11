import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, d, eps):
        super(LayerNormalization, self).__init__()
        self.d = d
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(self.d))
        self.bias = nn.Parameter(torch.zeros(self.d))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = self.alpha * x + self.bias
        return x
    
def layerNormalization(input_size,eps=1e-6):
    return LayerNormalization(input_size,eps)