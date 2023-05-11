import torch
import torch.nn as nn

class ClippedReLU(torch.nn.Module):
    def __init__(self,ceiling=10.0):
        super(ClippedReLU, self).__init__()
        self.ceiling = ceiling

    def forward(self, x):
        return torch.max(torch.max(torch.zeros_like(x), x), torch.full(x.shape, self.ceiling))
    
class ActiveFunction(nn.Module):
    def __init__(self, active_function_name,ceiling=10.0):
        super(ActiveFunction, self).__init__()
        assert active_function_name in ['relu', 'sigmoid', 'tanh', 'softmax','leakyRelu','clippedrelu','swish','gelu']
        if active_function_name == 'relu':
            self.active_function = nn.ReLU()
        elif active_function_name == 'sigmoid':
            self.active_function = nn.Sigmoid()
        elif active_function_name == 'tanh':
            self.active_function = nn.Tanh()
        elif active_function_name == 'softmax':
            self.active_function = nn.Softmax()
        elif active_function_name == 'leakyRelu':
            self.active_function = nn.LeakyReLU()
        elif active_function_name == 'clippedrelu':
            self.active_function = ClippedReLU(ceiling)
        elif active_function_name == 'swish':
            self.active_function = nn.SiLU()
        elif active_function_name == 'gelu':
            self.active_function = nn.GELU()
        else:
            raise ValueError('Unknown active function name: {}'.format(active_function_name))

    def forward(self, x):
        x = self.active_function(x)
        return x
    
def activeFunctionLayer(active_function_name='relu',ceiling=10.0):
    return ActiveFunction(active_function_name,ceiling)