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
    """Mains windows and anchor-point appliance targets across several houses.

    All houses are concatenated into one array; window starts are global
    offsets into it, stored house by house. Windows never cross a segment
    boundary, so they never cross a house boundary either.

    Arrays stay in watts. Items are normalized on the way out once
    ``normalizer`` is set (fit it on the training dataset, then assign the
    same instance to the validation and test datasets); until then they are
    returned in watts.

    Index with a single window id or with a sequence of ids; the latter
    returns a whole batch with one vectorised gather, which is how
    :func:`make_window_loader` uses it.
    """

    def __init__(
        self,
        houses: Sequence[HouseSeries],
        input_length: int,
        drop_target_above_aggregate: bool = True,
        normalizer: Optional[Normalizer] = None,
    ) -> None:
        if not houses:
            raise ValueError("no houses given")

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
        if len(self.starts) == 0:
            raise ValueError(f"no house has a valid window of length {input_length}")
        self.aggregate = np.concatenate([house.aggregate for house in houses])
        self.target = np.concatenate([house.target for house in houses])
        self.input_length = input_length
        self.normalizer = normalizer
        self._window = np.arange(input_length)
        self._anchor = anchor_offset(input_length)

    @classmethod
    def from_files(cls, files: Sequence[Union[str, Path]], target: str, **kwargs) -> "NILMWindowDataset":
        return cls([load_house(file, target) for file in files], **kwargs)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.starts[index]
        x = self.aggregate[np.asarray(starts)[..., None] + self._window]
        y = self.target[starts + self._anchor]
        if self.normalizer is not None:
            x = self.normalizer.normalize_input(x)
            y = self.normalizer.normalize_target(y)
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
    max_per_house: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, HouseBalancedSampler]:
    """Return a batched loader and its sampler (call ``sampler.set_epoch`` each epoch)."""

    sampler = HouseBalancedSampler(dataset.windows_per_house, max_per_house, shuffle, seed)
    loader = DataLoader(
        dataset,
        # The dataset gathers whole batches itself, so automatic batching is off.
        sampler=BatchSampler(sampler, batch_size, drop_last=False),
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, sampler
