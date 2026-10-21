"""Load split CSVs as NumPy arrays."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HouseSeries:
    """One house's aligned mains, appliance and segment-id columns."""

    name: str
    aggregate: np.ndarray  # float32 mains power, watts
    target: np.ndarray  # float32 appliance power, watts
    segment: np.ndarray  # int64 contiguous-recording id


def split_files(root: Union[str, Path], target: str, split: str) -> list[Path]:
    """Return the sorted house CSVs of ``<root>/<target>/<split>``."""

    split_dir = Path(root) / target / split
    files = sorted(split_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSVs found in {split_dir}")
    return files


def load_house(csv_path: Union[str, Path], target: str) -> HouseSeries:
    """Load one split CSV's ``Aggregate``, ``target`` and ``Segment`` columns."""

    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path, usecols=["Aggregate", target, "Segment"])
    return HouseSeries(
        name=csv_path.stem,
        aggregate=frame["Aggregate"].to_numpy(dtype=np.float32),
        target=frame[target].to_numpy(dtype=np.float32),
        segment=frame["Segment"].to_numpy(dtype=np.int64),
    )
