"""Compose run configs from configs/base.yaml and the selected model file."""

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def appliance_slug(target: str) -> str:
    """Map a target column name to its run directory name, e.g. 'Washing Machine' -> 'washing_machine'."""

    return target.lower().replace("-", "_").replace(" ", "_")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``base`` with ``override`` applied; nested dicts merge, everything else replaces."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def load_config(model: Optional[str] = None, target: Optional[str] = None) -> Dict[str, Any]:
    """Merge base -> models/<model>.yaml; ``model`` and ``target`` default to base.yaml's selection."""

    config = _read(CONFIG_DIR / "base.yaml")
    if target is not None:
        config["target"] = target
    return deep_merge(config, _read(CONFIG_DIR / "models" / f"{model or config['model']}.yaml"))


def load_config_from_cli(description: str) -> Dict[str, Any]:
    """``load_config`` with optional ``--model``, ``--target`` and ``--seed`` overrides of base.yaml."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", help="configs/models/<model>.yaml")
    parser.add_argument("--target", help="appliance column, e.g. Kettle")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.model, args.target)
    if args.seed is not None:
        config["seed"] = args.seed
    return config
