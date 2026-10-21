"""Run directory I/O: config, best checkpoint, metrics and result."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from src.utils.config import appliance_slug

RUNS_DIR = Path("runs")
BEST = "best.pt"
CONFIG = "config.yaml"
METRICS = "metrics.csv"
RESULT = "result.json"
METRIC_FIELDS = ("epoch", "train_loss", "val_loss", "is_best", "seconds")


def run_dir_path(model: str, target: str, run_name: Optional[str] = None) -> Path:
    """``runs/<model>/<appliance>/<run name>``; the name defaults to a timestamp."""

    return RUNS_DIR / model / appliance_slug(target) / (run_name or datetime.now().strftime("%Y%m%d-%H%M%S"))


def create_run_dir(run_dir: Path) -> None:
    """Create a fresh run directory; refuse to reuse one so results are never overwritten."""

    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists; choose another name")
    run_dir.mkdir(parents=True)


def save_config(run_dir: Path, config: Dict[str, Any]) -> None:
    (run_dir / CONFIG).write_text(yaml.safe_dump(config, sort_keys=False))


def read_config(run_dir: Path) -> Dict[str, Any]:
    return yaml.safe_load((run_dir / CONFIG).read_text())


def save_result(run_dir: Path, result: Dict[str, Any]) -> None:
    (run_dir / RESULT).write_text(json.dumps(result, indent=2))


def read_result(run_dir: Path) -> Optional[Dict[str, Any]]:
    """The run's result.json, or None if the run never finished."""

    path = run_dir / RESULT
    return json.loads(path.read_text()) if path.is_file() else None


def save_best(run_dir: Path, *, model, config, normalizer, epoch: int, val_loss: float) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "target": config["target"],
            "normalizer": normalizer.to_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        run_dir / BEST,
    )


def append_metrics(run_dir: Path, row: Dict[str, Any]) -> None:
    path = run_dir / METRICS
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
