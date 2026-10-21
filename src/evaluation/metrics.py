"""Disaggregation metrics on appliance power in watts."""

from typing import Dict

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in watts."""

    return float(np.mean(np.abs(y_pred - y_true)))


def sae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Signal aggregate error: relative error of total energy."""

    total = np.sum(y_true, dtype=np.float64)
    return float(abs(np.sum(y_pred, dtype=np.float64) - total) / total)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """All metrics as a dict."""

    return {"mae": mae(y_true, y_pred), "sae": sae(y_true, y_pred)}
