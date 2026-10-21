"""Per-epoch window sampling that balances houses and mixes them in every batch."""

from typing import Iterator, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


class HouseBalancedSampler(Sampler[int]):
    """Draw up to ``max_per_house`` windows from each house, then shuffle them together.

    Indices refer to a dataset whose windows are stored house by house, with
    ``windows_per_house[i]`` windows for house ``i``. The draw is seeded by
    ``(seed, epoch)``: call :meth:`set_epoch` before each training epoch to get
    a fresh subset and order, while reruns stay reproducible.

    With ``shuffle=False`` (validation/test) the subset is fixed and each
    house's windows stay in time order.
    """

    def __init__(
        self,
        windows_per_house: Sequence[int],
        max_per_house: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if max_per_house is not None and max_per_house < 1:
            raise ValueError("max_per_house must be positive or None")
        self.windows_per_house = [int(count) for count in windows_per_house]
        self.max_per_house = max_per_house
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _take(self, count: int) -> int:
        return count if self.max_per_house is None else min(count, self.max_per_house)

    def __len__(self) -> int:
        return sum(self._take(count) for count in self.windows_per_house)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng((self.seed, self.epoch))
        chosen = []
        offset = 0
        for count in self.windows_per_house:
            take = self._take(count)
            if take < count:
                indices = np.sort(rng.choice(count, take, replace=False))
            else:
                indices = np.arange(count)
            chosen.append(indices + offset)
            offset += count
        indices = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
        if self.shuffle:
            rng.shuffle(indices)
        return iter(indices.tolist())
