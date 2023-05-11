import torch.nn as nn
import torch


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, hidden, OutputMode):
        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.hidden = hidden
        self.OutputMode = OutputMode
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout,
                            bidirectional=True, batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input):
        if self.hidden is None:
            h_t = torch.zeros(2*self.num_layers, input.size(0),
                              self.hidden_dim).to(input.device)
            c_t = torch.zeros(2*self.num_layers, input.size(0),
                              self.hidden_dim).to(input.device)
        else:
            h_t, c_t = self.hidden

        out, (hn, cn) = self.lstm(input, (h_t, c_t))
        out = self.fc(torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1))
        if self.OutputMode == 'sequence':
            return self.fc(hn)
        elif self.OutputMode == 'last':
            return out


def bilstmLayer(input_size, hidden_size, output_size, num_layers=2, dropout=0.2, CellState=[], HiddenState=[], OutputMode='last'):
    if CellState != [] and HiddenState != []:
        hidden = (HiddenState, CellState)
    else:
        hidden = None

    model = BiLSTM(input_size, hidden_size, output_size,
                   num_layers, dropout, hidden, OutputMode)
    return model
