"""Sequence-to-point window selection."""

import numpy as np


def anchor_offset(input_length: int) -> int:
    """Offset of the predicted sample inside a window (floor midpoint)."""

    return input_length // 2


def segment_runs(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(starts, stops)`` of each run of consecutive equal segment ids."""

    breaks = np.flatnonzero(np.diff(segment) != 0) + 1
    return np.concatenate(([0], breaks)), np.concatenate((breaks, [len(segment)]))


def window_starts(segment: np.ndarray, input_length: int) -> np.ndarray:
    """Start index of every window lying entirely inside one segment run."""

    return np.concatenate(
        [
            np.arange(start, stop - input_length + 1, dtype=np.int64)
            for start, stop in zip(*segment_runs(segment))
            if stop - start >= input_length
        ]
    )


def valid_window_starts(
    segment: np.ndarray,
    aggregate: np.ndarray,
    target: np.ndarray,
    input_length: int,
    drop_target_above_aggregate: bool,
) -> np.ndarray:
    """All usable window starts of one house, in time order."""

    starts = window_starts(segment, input_length)
    if drop_target_above_aggregate:
        # A sub-meter reading above mains is a metering artefact.
        anchors = starts + anchor_offset(input_length)
        starts = starts[target[anchors] <= aggregate[anchors]]
    return starts
