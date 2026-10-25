"""Train the model and appliance selected in configs/base.yaml into runs/<model>/<appliance>/<seed>.

Example:
    python -m scripts.train
"""

from __future__ import annotations

from src.data import Normalizer
from src.training.checkpoint import run_dir_path
from src.training.trainer import load_split, train_run
from src.utils.config import load_config


def main() -> None:
    config = load_config()
    train_set = load_split(config, "train")
    val_set = load_split(config, "val")
    # Statistics come from the training split only and are shared with validation.
    normalizer = Normalizer.fit(*train_set.anchor_values())
    run_dir = run_dir_path(config["model"]["name"], config["target"], str(config["seed"]))
    train_run(config, run_dir, train_set, val_set, normalizer)


if __name__ == "__main__":
    main()
