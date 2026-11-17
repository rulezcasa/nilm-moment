"""Evaluate runs/<model>/<appliance>/<seed> selected in configs/base.yaml and log it to W&B.

Example:
    python -m scripts.evaluate
    python -m scripts.evaluate --model moment_frozen --target Kettle --seed 43
"""

from __future__ import annotations

from src.evaluation import evaluate_run
from src.training.checkpoint import run_dir_path
from src.utils.config import load_config_from_cli


def main() -> None:
    config = load_config_from_cli("Evaluate one trained run.")
    run_dir = run_dir_path(config["model"]["name"], config["target"], str(config["seed"]))
    evaluate_run(run_dir, config["test"]["split"])


if __name__ == "__main__":
    main()
