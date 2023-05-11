import torch
import torch.nn as nn


class LSTNet(nn.Module):
    def __init__(self, input_size, channel, hidden_size, skip_size, output_size, num_layers, kernel_size, dropout):
        super(LSTNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.skip_size = skip_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.channel=channel

        # Convolutional Layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=self.channel,
                      out_channels=hidden_size, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=hidden_size,
                      out_channels=hidden_size, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # Recurrent Layers
        self.lstm_layers = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

        # Skip Connection Layer
        self.skip_layer = nn.Sequential(
            nn.Linear(in_features=skip_size*hidden_size,
                      out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # Output Layer
        self.output_layer = nn.Linear(
            in_features=hidden_size*2, out_features=output_size)

    def forward(self, x):
        # Convolutional Layers
        conv_output = self.conv_layers(x)
        # Recurrent Layers
        lstm_output, _ = self.lstm_layers(conv_output)

        # Skip Connection Layer
        skip_output = lstm_output[:, -self.skip_size:, :]
        skip_output = skip_output.permute(0, 2, 1).contiguous()
        skip_output = skip_output.view(-1, self.skip_size*self.hidden_size)
        skip_output = self.skip_layer(skip_output)

        # Final Output Layer
        final_output = lstm_output[:, -1, :]

        # Combine Skip and Final Output
        output = torch.cat([skip_output, final_output], dim=1)

        # Output Layer
        output = self.output_layer(output)

        return output


def lstnetLayer(input_size, channel, hidden_size,  output_size, skip_size=5, num_layers=2, kernel_size=1, dropout=0.2):
    return LSTNet(input_size, channel, hidden_size, skip_size, output_size, num_layers, kernel_size, dropout)
