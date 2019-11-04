
from torch import nn
from models.modules import BaseModule


class LSTM(BaseModule):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            output_size=1,
            dropout=0.0):
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
        out, _ = self.lstm(x)
        out = self.linear(out.tanh())

        return out
