"""Streaming sliding-window dataset for the generated NILM split CSVs."""

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info


class NILMWindowDataset(IterableDataset):
    """Yield 512-sample mains windows and midpoint appliance targets.

    Files are read one at a time, which avoids concatenating the very large
    house files into memory. ``max_windows_per_file`` is useful for balanced
    experiments because otherwise the largest houses dominate each epoch.
    """

    def __init__(
        self,
        files: Sequence[Union[str, Path]],
        target: str,
        input_length: int = 512,
        input_mean: float = 0.0,
        input_std: float = 1.0,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        max_windows_per_file: Optional[int] = 100_000,
        shuffle: bool = False,
        drop_target_above_aggregate: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not files:
            raise ValueError("no split CSV files were found")
        if input_length < 1 or input_std <= 0 or target_std <= 0:
            raise ValueError("invalid window length or normalization standard deviation")
        self.files = tuple(Path(file) for file in files)
        self.target = target
        self.input_length = input_length
        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.max_windows_per_file = max_windows_per_file
        self.shuffle = shuffle
        self.drop_target_above_aggregate = drop_target_above_aggregate
        self.seed = seed

    def _file_windows(self, frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        starts: list[np.ndarray] = []
        for _, segment in frame.groupby("Segment", sort=False):
            start, stop = segment.index[0], segment.index[-1] + 1
            if stop - start >= self.input_length:
                starts.append(np.arange(start, stop - self.input_length + 1, dtype=np.int64))
        if not starts:
            return np.empty(0, dtype=np.int64)
        indices = np.concatenate(starts)
        if self.drop_target_above_aggregate:
            midpoint = indices + self.input_length // 2
            valid = frame.loc[midpoint, self.target].to_numpy() <= frame.loc[midpoint, "Aggregate"].to_numpy()
            indices = indices[valid]
        if self.max_windows_per_file and len(indices) > self.max_windows_per_file:
            indices = rng.choice(indices, self.max_windows_per_file, replace=False)
        if self.shuffle:
            rng.shuffle(indices)
        return indices

    def __iter__(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        files = self.files[worker_id::worker_count]
        rng = np.random.default_rng(self.seed + worker_id)
        if self.shuffle:
            files = list(files)
            rng.shuffle(files)

        for file in files:
            frame = pd.read_csv(
                file,
                usecols=["Aggregate", self.target, "Segment"],
            ).reset_index(drop=True)
            starts = self._file_windows(frame, rng)
            aggregate = frame["Aggregate"].to_numpy(dtype=np.float32)
            target = frame[self.target].to_numpy(dtype=np.float32)
            for start in starts:
                end = int(start) + self.input_length
                x = (aggregate[start:end] - self.input_mean) / self.input_std
                y = (target[int(start) + self.input_length // 2] - self.target_mean) / self.target_std
                yield torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
