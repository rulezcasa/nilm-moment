"""Train a NILM sequence-to-point model on one appliance.

Examples:
    python -m scripts.train --model cnn --target Kettle
    python -m scripts.train --model cnn --target Kettle --run-name baseline
"""

from __future__ import annotations

import argparse

from src.data import Normalizer
from src.training.checkpoint import run_dir_path
from src.training.trainer import load_split, train_run
from src.utils.config import available_models, load_config

TARGETS = ("Kettle", "Fridge-Freezer", "Washing Machine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn", choices=available_models())
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--run-name", help="defaults to a timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.model, args.target)
    train_set = load_split(config, "train")
    val_set = load_split(config, "val")
    # Statistics come from the training split only and are shared with validation.
    normalizer = Normalizer.fit(*train_set.anchor_values())
    train_run(config, run_dir_path(args.model, args.target, args.run_name), train_set, val_set, normalizer)


if __name__ == "__main__":
    main()
