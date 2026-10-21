"""Grid hyperparameter search for one model on one appliance.

Examples:
    python -m scripts.search --model cnn --target Kettle --name v1
    python -m scripts.search --model cnn --target Kettle --name v1 --retrain
"""

from __future__ import annotations

import argparse
from datetime import datetime

from src.training.checkpoint import run_dir_path
from src.training.search import run_search
from src.utils.config import available_models, load_config

TARGETS = ("Kettle", "Fridge-Freezer", "Washing Machine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn", choices=available_models())
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--name", help="search name; defaults to a timestamp, reuse it to continue a search")
    parser.add_argument("--retrain", action="store_true", help="after the trials, train the best parameters on the full budget")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.model, args.target)
    name = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_search(config, run_dir_path(args.model, args.target, f"search-{name}"), retrain=args.retrain)


if __name__ == "__main__":
    main()
