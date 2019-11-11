
from torch import nn
from models.modules import BaseModule


class LSTM(BaseModule):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            output_size=1,
            dropout_in=0.0,
            dropout_lstm=0.0,
            dropout_linear=0.0):
        super(LSTM, self).__init__()

        self.dropout_in = nn.Dropout(dropout_in)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_lstm)

        self.dropout_linear = nn.Dropout(dropout_linear)

        self.linear = nn.Linear(
            hidden_size,
            output_size)

    def forward(self, x):
        out = self.dropout_in(x)
        out, _ = self.lstm(out)
        out = self.dropout_linear(out)
        out = self.linear(out.tanh())

        return out
