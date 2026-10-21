"""Compose run configs from configs/base.yaml, a model file and an appliance file."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def available_models(config_dir: Path = CONFIG_DIR) -> list[str]:
    return sorted(path.stem for path in (config_dir / "models").glob("*.yaml"))


def appliance_slug(target: str) -> str:
    """Map a target column name to its config filename, e.g. 'Washing Machine' -> 'washing_machine'."""

    return target.lower().replace("-", "_").replace(" ", "_")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``base`` with ``override`` applied; nested dicts merge, everything else replaces."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def load_config(model: str, target: str, config_dir: Union[str, Path] = CONFIG_DIR) -> Dict[str, Any]:
    """Merge base -> models/<model>.yaml -> appliances/<target>.yaml."""

    config_dir = Path(config_dir)
    config = _read(config_dir / "base.yaml")
    config = deep_merge(config, _read(config_dir / "models" / f"{model}.yaml"))
    config = deep_merge(config, _read(config_dir / "appliances" / f"{appliance_slug(target)}.yaml"))
    if config.get("target") != target:
        raise ValueError(f"appliance config declares target {config.get('target')!r}, expected {target!r}")
    return config
