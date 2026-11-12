"""Grid hyperparameter search for the model and appliance selected in configs/base.yaml.

Example:
    python -m scripts.search
    python -m scripts.search --model moment_frozen --target Kettle --seed 43
"""

from __future__ import annotations

from src.training.checkpoint import run_dir_path
from src.training.search import run_search
from src.utils.config import load_config_from_cli


def main() -> None:
    config = load_config_from_cli("Grid hyperparameter search.")
    run_search(config, run_dir_path(config["model"]["name"], config["target"], f"search-{config['search']['name']}"))


if __name__ == "__main__":
    main()
