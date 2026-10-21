"""Per-epoch window sampling that balances houses and mixes them in every batch."""

from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class HouseBalancedSampler(Sampler[int]):
    """Draw up to ``max_per_house`` windows from each house, then shuffle them together."""

    def __init__(self, windows_per_house: Sequence[int], max_per_house: int, shuffle: bool, seed: int) -> None:
        self.windows_per_house = list(windows_per_house)
        self.max_per_house = max_per_house
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return sum(min(count, self.max_per_house) for count in self.windows_per_house)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng((self.seed, self.epoch))
        chosen = []
        offset = 0
        for count in self.windows_per_house:
            take = min(count, self.max_per_house)
            indices = np.sort(rng.choice(count, take, replace=False)) if take < count else np.arange(count)
            chosen.append(indices + offset)
            offset += count
        indices = np.concatenate(chosen)
        if self.shuffle:
            rng.shuffle(indices)
        return iter(indices.tolist())
