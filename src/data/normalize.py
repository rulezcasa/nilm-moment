"""Z-score normalization fitted on the training split.

Fit once on the training windows and reuse the same statistics for
validation and test; fitting on held-out houses would leak them into
training. The fitted values travel inside each checkpoint.
"""

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Normalizer:
    """Mean/std of mains input and appliance target, in watts."""

    input_mean: float
    input_std: float
    target_mean: float
    target_std: float

    def __post_init__(self) -> None:
        if not (self.input_std > 0 and self.target_std > 0):
            raise ValueError(f"normalization standard deviations must be positive: {self}")

    @classmethod
    def fit(cls, aggregate: np.ndarray, target: np.ndarray) -> "Normalizer":
        """Fit from the mains and appliance values at the training anchors.

        Use the anchor of every valid training window (see
        ``NILMWindowDataset.anchor_values``): that is exactly the target
        distribution the model regresses, and since windows overlap almost
        every mains sample is some window's anchor.
        """

        return cls(
            input_mean=float(aggregate.mean(dtype=np.float64)),
            input_std=float(aggregate.std(dtype=np.float64)),
            target_mean=float(target.mean(dtype=np.float64)),
            target_std=float(target.std(dtype=np.float64)),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, float]) -> "Normalizer":
        return cls(**values)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def normalize_input(self, x):
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        return (y - self.target_mean) / self.target_std

    def target_to_watts(self, y):
        """Invert :meth:`normalize_target`, e.g. on model predictions."""

        return y * self.target_std + self.target_mean
