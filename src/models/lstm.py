import torch
from torch import nn
from models.modules import BaseModule
from models.multilinear import MultiLinear


class LSTM(BaseModule):
    """LSTM layer.

    Shapes
    ----------
    - Input:  (batch, sequence, input_size)
    - Output: (batch, sequence, output_size)

    Parameters
    ----------
    num_dynamic: int
        The number of dynamic features.
    num_static: int
        The number of static features.
    lstm_hidden_size: int
        The number of nodes per LSTM layer.
    lstm_num_layers: int
        The number of LSTM layers.
    dense_hidden_size: int
        The number of nodes per dense layer.
    dense_num_layers: int
        The number of dense layers.
    dense_activation: torch.nn.modules.activation
        The activation function used for dense layers.
    output_size: int (default: 1)
        The output size.
    dropout_in: float (default: 0.0)
        Dropout applied to the input data, [0, 1).
    dropout_lstm: float (default: 0.0)
        Dropout applied between LSTM layers, [0, 1).
    dropout_linear: float (default: 0.0)
        Dropout applied after dense layers, [0, 1).
    """

    def __init__(
            self,
            num_dynamic,
            num_static,
            lstm_hidden_size,
            lstm_num_layers,
            dense_hidden_size,
            dense_num_layers,
            dense_activation,
            output_size=1,
            dropout_in=0.0,
            dropout_lstm=0.0,
            dropout_linear=0.0):
        super(LSTM, self).__init__()

        self.dropout_in = nn.Dropout(dropout_in)

        self.lstm = nn.LSTM(
            input_size=num_dynamic,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_lstm)

        self.dense = MultiLinear(
            input_size=num_static + lstm_hidden_size,
            hidden_size=dense_hidden_size,
            output_size=1,
            num_layers=dense_num_layers,
            dropout=dropout_linear,
            activation=dense_activation
        )

    def forward(self, d: torch.Tensor, s: torch.Tensor):

        lstm_out, _ = self.lstm(self.dropout_in(d))

        out = torch.cat(
            (lstm_out, s.expand(lstm_out.size(0), lstm_out.size(1), s.size(2))), -1
        )

        return self.dense(out)
