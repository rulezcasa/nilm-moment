"""Train a NILM sequence-to-point model on one appliance.

Each run writes to runs/<model>/<appliance>/<run name>/; see
src/training/checkpoint.py for what goes in it.

Examples:
    python -m scripts.train --model cnn --target Kettle
    python -m scripts.train --model cnn --target Kettle --run-name baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import Normalizer
from src.training.checkpoint import RUNS_DIR, run_dir_path
from src.training.trainer import load_split, train_run
from src.utils.config import available_models, load_config

TARGETS = ("Kettle", "Fridge-Freezer", "Washing Machine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn", choices=available_models())
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--run-name", help="defaults to a timestamp")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-windows-per-file", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.model, args.target)
    if args.data_root:
        config["data"]["root"] = str(args.data_root)
    if args.max_windows_per_file is not None:
        config["training"]["max_windows_per_file"] = args.max_windows_per_file
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    run_dir = run_dir_path(args.runs_dir, args.model, args.target, args.run_name)
    if run_dir.exists():  # fail before spending time loading data
        raise FileExistsError(f"{run_dir} already exists; choose another --run-name")

    train_set = load_split(config, "train")
    val_set = load_split(config, "val")
    # Statistics come from the training split only and are shared with validation.
    normalizer = Normalizer.fit(*train_set.anchor_values())
    train_run(config, run_dir, train_set, val_set, normalizer)


if __name__ == "__main__":
    main()
