"""Random hyperparameter search for one model on one appliance.

Runs the ``search`` block of configs/models/<model>.yaml, then retrains the
best trial on the full budget; see src/training/search.py for the layout.
Stop it at any time (Ctrl+C) and rerun the same command to continue: finished
trials are skipped and an interrupted trial is retrained from scratch.

Examples:
    python -m scripts.search --model cnn --target Kettle
    python -m scripts.search --model cnn --target Kettle --name v1 --trials 8
    python -m scripts.search --model cnn --target Kettle --name v1 --trials 16   # continue, adding 8 more
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.training.checkpoint import RUNS_DIR
from src.training.search import run_search
from src.utils.config import appliance_slug, available_models, load_config

TARGETS = ("Kettle", "Fridge-Freezer", "Washing Machine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn", choices=available_models())
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--name", help="search name; defaults to a timestamp, reuse it to continue a search")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--trials", type=int, help="override search.trials; raising it on an existing search adds trials")
    parser.add_argument("--no-retrain", action="store_true", help="stop after the trials")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.model, args.target)
    if args.data_root:
        config["data"]["root"] = str(args.data_root)
    if args.trials is not None:
        config.setdefault("search", {})["trials"] = args.trials

    name = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    search_dir = args.runs_dir / args.model / appliance_slug(args.target) / f"search-{name}"
    run_search(config, search_dir, retrain=not args.no_retrain)


if __name__ == "__main__":
    main()
