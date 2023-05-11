import torch
import numpy as np

class Linear(torch.nn.Module):
    def __init__(self, input_size, output_size,WeightsInitializer,BiasInitializer):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        if WeightsInitializer == 'glorot' or WeightsInitializer == 'xavier':
            self.weight = torch.nn.Parameter(torch.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size)))
        elif WeightsInitializer == 'he':
            self.weight = torch.nn.Parameter(torch.randn(input_size, output_size) * np.sqrt(2 / input_size))
        elif WeightsInitializer == 'random':
            self.weight = torch.nn.Parameter(torch.randn(input_size, output_size))
        elif WeightsInitializer == 'orthogonal':
            self.weight = torch.nn.Parameter(torch.nn.init.orthogonal_(torch.empty(input_size, output_size)))
        elif WeightsInitializer == 'ones':
            self.weight = torch.nn.Parameter(torch.ones(input_size, output_size))
        elif WeightsInitializer == 'zeros':
            self.weight = torch.nn.Parameter(torch.zeros(input_size, output_size))
        
        if BiasInitializer == 'zeros':
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
        elif BiasInitializer == 'ones':
            self.bias = torch.nn.Parameter(torch.ones(output_size))
        elif BiasInitializer == 'random':
            self.bias = torch.nn.Parameter(torch.randn(output_size))

    def forward(self, input):
        if len(input.size()) == 3:
            batch_size, time_step, input_size = input.size()
            input = input.view(batch_size*time_step, input_size)
            output = torch.matmul(input, self.weight) + self.bias
            output = output.view(batch_size, time_step, self.output_size)
            return output
        elif len(input.size()) == 2:
            output = torch.matmul(input, self.weight) + self.bias
            return output
        else:
            print('Fully Connected Layer: input must be a 2D or 3D tensor')
            exit()
    

def fullyConnectedLayer(input_size, output_size,WeightsInitializer='glorot',BiasInitializer='zeros'):
    return Linear(input_size, output_size,WeightsInitializer,BiasInitializer)



