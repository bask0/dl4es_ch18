
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.0):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout)

        self.linear = nn.Linear(
            hidden_size,
            output_size)

    def forward(self, x):
        out = self.lstm(x)
        out = self.linear(out)

        return out
