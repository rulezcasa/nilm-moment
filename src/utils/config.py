"""Compose run configs from configs/base.yaml and the selected model file."""

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
