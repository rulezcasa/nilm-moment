"""Sequence-to-point window selection.

A window is ``input_length`` consecutive samples of one house; its anchor is
the sample at ``start + input_length // 2`` (index 256 of a 512 window), the
point whose appliance power the model predicts. Windows are identified by
their start index alone.
"""

import numpy as np


def anchor_offset(input_length: int) -> int:
    """Offset of the predicted sample inside a window (floor midpoint)."""

    return input_length // 2


def segment_runs(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(starts, stops)`` of each run of consecutive equal segment ids.

    Runs are found from where the id changes, so a segment id that reappears
    later in the file is treated as a separate run instead of being bridged.
    """

    if len(segment) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    breaks = np.flatnonzero(np.diff(segment) != 0) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(segment)]))
    return starts, stops


def window_starts(segment: np.ndarray, input_length: int) -> np.ndarray:
    """Start index of every window lying entirely inside one segment run."""

    if input_length < 1:
        raise ValueError("input_length must be positive")
    pieces = [
        np.arange(start, stop - input_length + 1, dtype=np.int64)
        for start, stop in zip(*segment_runs(segment))
        if stop - start >= input_length
    ]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def drop_target_above_aggregate(
    starts: np.ndarray,
    aggregate: np.ndarray,
    target: np.ndarray,
    input_length: int,
) -> np.ndarray:
    """Discard windows whose anchor reports more appliance power than mains.

    The EDA showed such samples are metering artefacts: a sub-meter cannot
    physically draw more than the whole house.
    """

    anchors = starts + anchor_offset(input_length)
    return starts[target[anchors] <= aggregate[anchors]]


def valid_window_starts(
    segment: np.ndarray,
    aggregate: np.ndarray,
    target: np.ndarray,
    input_length: int,
    drop_target_above_aggregate_windows: bool = True,
) -> np.ndarray:
    """All usable window starts of one house, in time order."""

    starts = window_starts(segment, input_length)
    if drop_target_above_aggregate_windows:
        starts = drop_target_above_aggregate(starts, aggregate, target, input_length)
    return starts
