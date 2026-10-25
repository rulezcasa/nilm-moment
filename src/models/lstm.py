"""Sequence-to-point LSTM adapted from Kelly & Knottenbelt, "Neural NILM" (arXiv:1507.06594)."""

from typing import Any, Mapping, Sequence

from torch import Tensor, nn

from src.data.windowing import anchor_offset


class LSTMSeq2Point(nn.Module):
    """Predict the target appliance power at the midpoint of a ``[batch, length]`` window."""

    def __init__(
        self,
        input_length: int,
        conv_channels: int,
        conv_kernel_size: int,
        lstm_hidden_sizes: Sequence[int],
        dense_units: int,
    ) -> None:
        super().__init__()
        first_hidden, second_hidden = lstm_hidden_sizes
        self.target_index = anchor_offset(input_length)
        # Asymmetric padding keeps the sequence length for even kernels.
        left = (conv_kernel_size - 1) // 2
        self.convolution = nn.Sequential(
            nn.ConstantPad1d((left, conv_kernel_size - 1 - left), 0.0),
            nn.Conv1d(1, conv_channels, conv_kernel_size),
        )
        self.lstm1 = nn.LSTM(conv_channels, first_hidden, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * first_hidden, second_hidden, batch_first=True, bidirectional=True)
        self.regressor = nn.Sequential(
            nn.Linear(2 * second_hidden, dense_units),
            nn.Tanh(),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        sequence = self.convolution(x.unsqueeze(1)).transpose(1, 2)
        sequence, _ = self.lstm1(sequence)
        sequence, _ = self.lstm2(sequence)
        return self.regressor(sequence[:, self.target_index, :])


def build_lstm(model_cfg: Mapping[str, Any], input_length: int) -> LSTMSeq2Point:
    """Construct the LSTM from a config's ``model`` section."""

    return LSTMSeq2Point(input_length, **{key: value for key, value in model_cfg.items() if key != "name"})
