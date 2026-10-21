"""Z-score normalization fitted on the training split."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Normalizer:
    """Mean/std of mains input and appliance target, in watts."""

    input_mean: float
    input_std: float
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, aggregate: np.ndarray, target: np.ndarray) -> "Normalizer":
        """Fit from the mains and appliance values at the training anchors."""

        return cls(
            input_mean=float(aggregate.mean(dtype=np.float64)),
            input_std=float(aggregate.std(dtype=np.float64)),
            target_mean=float(target.mean(dtype=np.float64)),
            target_std=float(target.std(dtype=np.float64)),
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def normalize_input(self, x):
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        return (y - self.target_mean) / self.target_std

    def denormalize_target(self, y):
        return y * self.target_std + self.target_mean
