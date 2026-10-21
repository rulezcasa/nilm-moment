"""Sequence-to-point CNN (Zhang et al., arXiv:1612.09106)."""

from typing import Any, Mapping, Sequence

from torch import Tensor, nn


class CNNSeq2Point(nn.Module):
    """Predict the target appliance power at the midpoint of a ``[batch, length]`` window."""

    def __init__(
        self,
        input_length: int,
        conv_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        dense_units: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels, length = 1, input_length
        for out_channels, kernel_size, stride in zip(conv_channels, kernel_sizes, strides):
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, stride), nn.ReLU()]
            in_channels = out_channels
            length = (length - kernel_size) // stride + 1
        self.convolution = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear(in_channels * length, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.regressor(self.flatten(self.convolution(x.unsqueeze(1))))


def build_cnn(model_cfg: Mapping[str, Any], input_length: int) -> CNNSeq2Point:
    """Construct the CNN from a config's ``model`` section."""

    return CNNSeq2Point(input_length, **{key: value for key, value in model_cfg.items() if key != "name"})


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
