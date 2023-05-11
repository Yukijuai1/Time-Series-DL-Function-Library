import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()

        self.hidden_size = hidden_size

        # Input gate weights and bias
        self.W_iz = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hz = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))

        # Reset gate weights and bias
        self.W_ir = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hr = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_r = nn.Parameter(torch.zeros(hidden_size))

        # New gate weights and bias
        self.W_in = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hn = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_n = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_t):
        # Compute reset gate
        r_t = torch.sigmoid(torch.matmul(x_t, self.W_ir) +
                            torch.matmul(h_t, self.W_hr) + self.b_r)

        # Compute update gate
        z_t = torch.sigmoid(torch.matmul(x_t, self.W_iz) +
                            torch.matmul(h_t, self.W_hz) + self.b_z)

        # Compute new gate
        n_t = torch.tanh(torch.matmul(x_t, self.W_in) + r_t *
                         torch.matmul(h_t, self.W_hn) + self.b_n)

        # Compute new hidden state
        h_t_new = (1 - z_t) * n_t + z_t * h_t

        return h_t_new


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, hidden,OutputMode, batch_first=True):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.OutputMode = OutputMode

        self.gru_cells = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                gru_cell = GRUCell(input_size, hidden_size)
            else:
                gru_cell = GRUCell(hidden_size, hidden_size)
            self.gru_cells.append(gru_cell)

        self.hidden_state = hidden

    def forward(self, x):
        # If hidden state is not provided, initialize with zeros
        if self.hidden_state is None:
            h_t = torch.zeros(self.num_layers, x.size(0),
                              self.hidden_size).to(x.device)
        else:
            h_t = self.hidden_state

        # Split inputs into sequence length and batch size dimensions
        if self.batch_first:
            x = x.transpose(0, 1)

        # Process each input timestep
        for i in range(x.size(0)):
            x_t = x[i]
            h_t_new = []

            # Process each layer
            for layer_idx, gru_cell in enumerate(self.gru_cells):
                gru_input = x_t if layer_idx == 0 else h_t_new[layer_idx-1]
                h_t_new.append(gru_cell(gru_input, h_t[layer_idx]))

            # Update hidden state
            h_t = torch.stack(h_t_new, dim=0)
        # Transpose output back to (batch_size, sequence_length, hidden_size) if batch_first is True
        if self.batch_first:
            h_t = h_t.transpose(0, 1)

        self.hidden_state = None
        if self.OutputMode == 'sequence':
            return h_t
        elif self.OutputMode == 'last':
            return h_t[:, -1, :]


def gruLayer(input_size, hidden_size, num_layers=2, HiddenState=[], OutputMode='last'):
    if HiddenState != []:
        hidden = HiddenState
    else:
        hidden = None

    return GRU(input_size, hidden_size, num_layers, hidden, OutputMode)
