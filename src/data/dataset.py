"""In-memory sequence-to-point dataset and its DataLoader."""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

from .io import HouseSeries, load_house
from .normalize import Normalizer
from .sampling import HouseBalancedSampler
from .windowing import anchor_offset, valid_window_starts


class NILMWindowDataset(Dataset):
    """Mains windows and anchor-point appliance targets across several houses."""

    def __init__(self, houses: Sequence[HouseSeries], input_length: int, drop_target_above_aggregate: bool) -> None:
        starts = []
        offset = 0
        for house in houses:
            house_starts = valid_window_starts(
                house.segment,
                house.aggregate,
                house.target,
                input_length,
                drop_target_above_aggregate,
            )
            starts.append(house_starts + offset)
            offset += len(house.aggregate)

        self.house_names = tuple(house.name for house in houses)
        self.windows_per_house = tuple(len(house_starts) for house_starts in starts)
        self.starts = np.concatenate(starts)
        self.aggregate = np.concatenate([house.aggregate for house in houses])
        self.target = np.concatenate([house.target for house in houses])
        self.normalizer: Optional[Normalizer] = None  # set before iterating
        self._window = np.arange(input_length)
        self._anchor = anchor_offset(input_length)

    @classmethod
    def from_files(cls, files: Sequence[Union[str, Path]], target: str, **kwargs) -> "NILMWindowDataset":
        return cls([load_house(file, target) for file in files], **kwargs)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.starts[index]
        x = self.normalizer.normalize_input(self.aggregate[np.asarray(starts)[..., None] + self._window])
        y = self.normalizer.normalize_target(self.target[starts + self._anchor])
        return torch.from_numpy(np.asarray(x, dtype=np.float32)), torch.as_tensor(y, dtype=torch.float32)

    def anchor_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Mains and appliance watts at every window's anchor, for fitting a :class:`Normalizer`."""

        anchors = self.starts + self._anchor
        return self.aggregate[anchors], self.target[anchors]

    def summary(self) -> str:
        counts = ", ".join(f"{name}={count:,}" for name, count in zip(self.house_names, self.windows_per_house))
        return f"{len(self):,} windows ({counts})"


def make_window_loader(
    dataset: NILMWindowDataset,
    batch_size: int,
    max_per_house: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, HouseBalancedSampler]:
    """Return a batched loader and its sampler (call ``sampler.set_epoch`` each epoch)."""

    sampler = HouseBalancedSampler(dataset.windows_per_house, max_per_house, shuffle, seed)
    loader = DataLoader(
        dataset,
        # The dataset gathers whole batches itself, so automatic batching is off.
        sampler=BatchSampler(sampler, batch_size, drop_last=False),
        batch_size=None,
        num_workers=num_workers,
    )
    return loader, sampler
