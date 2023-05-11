import torch
import numpy as np


class HardSigmoid(torch.nn.Module):
    def __init__(self):
        super(HardSigmoid, self).__init__()

    def forward(self, x):
        x = torch.where(x < -2.5, torch.zeros_like(x), x)
        x = torch.where((x >= -2.5) & (x <= 2.5), 0.2 * x + 0.5, x)
        x = torch.where(x > 2.5, torch.ones_like(x), x)
        return x


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, hidden, InputWeightsInitializer,BiasInitializer,StateActivationFunction, GateActivationFunction, OutputMode):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden = hidden
        self.OutputMode = OutputMode

        if InputWeightsInitializer == 'glorot' or InputWeightsInitializer == 'xavier':
            self.weight_ih = torch.nn.Parameter(
                torch.randn(4 * hidden_size, input_size) * np.sqrt(2 / (input_size + hidden_size)))
            self.weight_hh = torch.nn.Parameter(
                torch.randn(4 * hidden_size, hidden_size) * np.sqrt(2 / (hidden_size + hidden_size)))
        elif InputWeightsInitializer == 'he':
            self.weight_ih = torch.nn.Parameter(
                torch.randn(4 * hidden_size, input_size) * np.sqrt(2 / input_size))
            self.weight_hh = torch.nn.Parameter(
                torch.randn(4 * hidden_size, hidden_size) * np.sqrt(2 / hidden_size))
        elif InputWeightsInitializer == 'random':
            self.weight_ih = torch.nn.Parameter(
                torch.randn(4 * hidden_size, input_size))
            self.weight_hh = torch.nn.Parameter(
                torch.randn(4 * hidden_size, hidden_size))
        elif InputWeightsInitializer == 'orthogonal':
            self.weight_ih = torch.nn.Parameter(
                torch.nn.init.orthogonal_(torch.empty(4 * hidden_size, input_size)))
            self.weight_hh = torch.nn.Parameter(
                torch.nn.init.orthogonal_(torch.empty(4 * hidden_size, hidden_size)))
        elif InputWeightsInitializer == 'ones':
            self.weight_ih = torch.nn.Parameter(
                torch.ones(4 * hidden_size, input_size))
            self.weight_hh = torch.nn.Parameter(
                torch.ones(4 * hidden_size, hidden_size))
        elif InputWeightsInitializer == 'zeros':
            self.weight_ih = torch.nn.Parameter(
                torch.zeros(4 * hidden_size, input_size))
            self.weight_hh = torch.nn.Parameter(
                torch.zeros(4 * hidden_size, hidden_size))
            
        if BiasInitializer == 'unit-forget-gate':
            self.bias_ih = torch.nn.Parameter(torch.zeros(4 * hidden_size))
            self.bias_hh = torch.nn.Parameter(torch.zeros(4 * hidden_size))
        elif BiasInitializer == 'ones':
            self.bias_ih = torch.nn.Parameter(torch.ones(4 * hidden_size))
            self.bias_hh = torch.nn.Parameter(torch.ones(4 * hidden_size))
        elif BiasInitializer == 'random':
            self.bias_ih = torch.nn.Parameter(torch.randn(4 * hidden_size))
            self.bias_hh = torch.nn.Parameter(torch.randn(4 * hidden_size))

        if StateActivationFunction == 'tanh':
            self.StateActivationFunction = torch.tanh
        elif StateActivationFunction == 'softsign':
            self.StateActivationFunction = torch.nn.functional.softsign

        if GateActivationFunction == 'sigmoid':
            self.GateActivationFunction = torch.sigmoid
        elif GateActivationFunction == 'hard-sigmoid':
            self.GateActivationFunction = HardSigmoid()

    def forward(self, input):

        input = input.to(torch.float32)
        if self.hidden is None:
            h_t = torch.zeros(self.num_layers, input.size(0),
                              self.hidden_size).to(input.device)
            c_t = torch.zeros(self.num_layers, input.size(0),
                              self.hidden_size).to(input.device)
        else:
            h_t, c_t = self.hidden

        outputs = []
        for t in range(input.size(1)):
            x_t = input[:, t, :]
            gates = torch.matmul(x_t, self.weight_ih.t()) + self.bias_ih + torch.matmul(h_t[-1], self.weight_hh.t()) + self.bias_hh
            f_t, i_t, o_t, g_t = gates.chunk(4, dim=1)
            f_t = self.GateActivationFunction(f_t)
            i_t = self.GateActivationFunction(i_t)
            o_t = self.GateActivationFunction(o_t)
            g_t = self.StateActivationFunction(g_t)
            c1_t = f_t * c_t[-1] + i_t * g_t
            h1_t = o_t * self.StateActivationFunction(c1_t)
            c_t = torch.cat([c_t[1:], c1_t.unsqueeze(0)], dim=0)
            h_t = torch.cat([h_t[1:], h1_t.unsqueeze(0)], dim=0)
            outputs.append(h1_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        if self.OutputMode == 'sequence':
            return outputs
        elif self.OutputMode == 'last':
            return h_t


def lstmLayer(input_size,
              hidden_size,
              num_layers=2,
              CellState=[],
              HiddenState=[],
              InputWeightsInitializer='glorot',
              BiasInitializer='unit-forget-gate',
              StateActivationFunction='tanh',
              GateActivationFunction='sigmoid',
              OutputMode='sequence'):

    if CellState != [] and HiddenState != []:
        hidden = (HiddenState, CellState)
    else:
        hidden = None

    model = LSTM(input_size, hidden_size, num_layers, hidden,InputWeightsInitializer,BiasInitializer,StateActivationFunction, GateActivationFunction, OutputMode)
    return model
