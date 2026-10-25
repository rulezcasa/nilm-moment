"""Constant sanity baselines: score "always zero" and "always the train median" on a held-out split.

Examples:
    python -m src.evaluation.sanity
    python -m src.evaluation.sanity --target Kettle --kind median --split val
"""

import argparse
import json
from typing import Any, Dict

import numpy as np

from src.training.checkpoint import run_dir_path
from src.training.trainer import load_split
from src.utils.config import load_config

from .metrics import compute_metrics

TARGETS = ("Kettle", "Fridge-Freezer", "Washing Machine")
KINDS = ("zero", "median")


def evaluate_constant(target: str, kind: str, split: str = "test") -> Dict[str, Any]:
    """Score a constant prediction on ``split`` and write ``runs/<kind>/<appliance>/baseline/eval_<split>.json``."""

    # Model config only fixes the shared data settings, so windows match the trained models.
    config = load_config("cnn", target)
    if kind == "zero":
        value = 0.0
    else:
        # Median of the train targets, never the evaluated split, to avoid leakage.
        value = float(np.median(load_split(config, "train").anchor_values()[1]))

    dataset = load_split(config, split)
    y_true = dataset.anchor_values()[1].astype(np.float64)
    y_pred = np.full_like(y_true, value)

    bounds = np.cumsum(dataset.windows_per_house)[:-1]
    houses = [
        {"house": name, "windows": count, **compute_metrics(true, pred)}
        for name, count, true, pred in zip(
            dataset.house_names, dataset.windows_per_house, np.split(y_true, bounds), np.split(y_pred, bounds)
        )
    ]
    result = {"split": split, "kind": kind, "value": value, **compute_metrics(y_true, y_pred), "houses": houses}

    run_dir = run_dir_path(kind, target, "baseline")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"eval_{split}.json").write_text(json.dumps(result, indent=2))
    print(f"{run_dir} {split}: value={value:.3f} W mae={result['mae']:.3f} W sae={result['sae']:.4f}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", nargs="+", default=list(TARGETS), choices=TARGETS)
    parser.add_argument("--kind", nargs="+", default=list(KINDS), choices=KINDS)
    parser.add_argument("--split", default="test", choices=("test", "val"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for target in args.target:
        for kind in args.kind:
            evaluate_constant(target, kind, args.split)


if __name__ == "__main__":
    main()
