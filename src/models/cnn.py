"""CNN sequence-to-point model for NILM.

The architecture in configs/models/cnn.yaml follows Figure 1 of Zhang et al.,
"Sequence-to-point learning with neural networks for non-intrusive load
monitoring" (arXiv:1612.09106), adapted from a 599-sample window to the
project's 512-sample MOMENT window.
"""

from dataclasses import dataclass, fields
from typing import Any, Mapping, Tuple

from torch import Tensor, nn


@dataclass(frozen=True)
class CNNConfig:
    """Architecture and regularization settings for :class:`CNNSeq2Point`.

    Values come from the ``model`` section of configs/models/cnn.yaml; there
    are deliberately no architecture defaults here.
    """

    input_length: int
    conv_channels: Tuple[int, ...]
    kernel_sizes: Tuple[int, ...]
    strides: Tuple[int, ...]
    dense_units: int
    dropout: float
    in_channels: int = 1
    output_dim: int = 1

    @classmethod
    def from_config(cls, model_cfg: Mapping[str, Any], input_length: int) -> "CNNConfig":
        """Build from a config's ``model`` section; unknown keys raise so typos surface."""

        values = {key: value for key, value in model_cfg.items() if key != "name"}
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown CNN config keys: {sorted(unknown)}")
        for key in ("conv_channels", "kernel_sizes", "strides"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(input_length=input_length, **values)

    def __post_init__(self) -> None:
        if self.input_length < 1:
            raise ValueError("input_length must be positive")
        if len(self.conv_channels) != len(self.kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have equal length")
        if len(self.strides) != len(self.kernel_sizes):
            raise ValueError("strides and kernel_sizes must have equal length")
        if any(k < 1 for k in self.kernel_sizes):
            raise ValueError("kernel sizes must be positive")
        if any(s < 1 for s in self.strides):
            raise ValueError("strides must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class CNNSeq2Point(nn.Module):
    """Predict the target appliance power at the window midpoint.

    Accepted input shapes are ``[batch, 512]`` and ``[batch, 1, 512]``.
    For an even window, the target index is ``input_length // 2`` (index 256
    for the project's 512-sample windows), matching the paper's floor-based
    midpoint definition.
    """

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_channels = self.config.in_channels
        for out_channels, kernel_size, stride in zip(
            self.config.conv_channels,
            self.config.kernel_sizes,
            self.config.strides,
        ):
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, stride),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        self.convolution = nn.Sequential(*layers)

        feature_length = self._feature_length()
        if feature_length < 1:
            raise ValueError("convolution stack is longer than input_length")
        self.flatten = nn.Flatten()
        self.regressor = nn.Sequential(
            nn.Linear(in_channels * feature_length, self.config.dense_units),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.dense_units, self.config.output_dim),
        )

    def _feature_length(self) -> int:
        length = self.config.input_length
        for kernel_size, stride in zip(self.config.kernel_sizes, self.config.strides):
            length = (length - kernel_size) // stride + 1
        return length

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError("expected input shape [batch, length] or [batch, channels, length]")
        if x.shape[1] != self.config.in_channels or x.shape[2] != self.config.input_length:
            raise ValueError(
                f"expected [batch, {self.config.in_channels}, {self.config.input_length}], "
                f"got {tuple(x.shape)}"
            )
        return self.regressor(self.flatten(self.convolution(x)))


def build_cnn(model_cfg: Mapping[str, Any], input_length: int) -> CNNSeq2Point:
    """Construct the CNN from a config's ``model`` section."""

    return CNNSeq2Point(CNNConfig.from_config(model_cfg, input_length))


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
