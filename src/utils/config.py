"""Compose run configs from configs/base.yaml, a model file and an appliance file."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def available_models() -> list[str]:
    return sorted(path.stem for path in (CONFIG_DIR / "models").glob("*.yaml"))


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
            merged[key] = value
    return merged


def _read(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def load_config(model: str, target: str) -> Dict[str, Any]:
    """Merge base -> models/<model>.yaml -> appliances/<target>.yaml."""

    config = _read(CONFIG_DIR / "base.yaml")
    config = deep_merge(config, _read(CONFIG_DIR / "models" / f"{model}.yaml"))
    return deep_merge(config, _read(CONFIG_DIR / "appliances" / f"{appliance_slug(target)}.yaml"))
