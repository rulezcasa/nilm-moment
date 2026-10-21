"""Weights & Biases logging, one W&B run per run directory."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import wandb

LOGGED_MODEL_JOBS = ("train", "search-final")


@dataclass(frozen=True)
class TrackingInfo:
    """How a run is labelled in W&B; built by the caller that knows the run's role."""

    name: str
    group: str
    job_type: str = "train"
    meta: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


def start_run(config: Dict[str, Any], run_dir: Path, info: TrackingInfo):
    settings = config["wandb"]
    run = wandb.init(
        project=settings["project"],
        entity=settings["entity"],
        mode=settings["mode"],
        dir=str(run_dir),
        name=info.name,
        group=info.group,
        job_type=info.job_type,
        tags=[config["model"]["name"], config["target"], info.job_type],
        config={**config, "meta": info.meta},
    )
    run.define_metric("epoch")
    run.define_metric("*", step_metric="epoch")
    run.define_metric("val_loss", summary="min")
    for key, rows in info.tables.items():
        run.log({key: wandb.Table(columns=list(rows[0]), data=[list(row.values()) for row in rows])})
    return run


def upload_model(run, path: Path, name: str, metadata: Dict[str, Any]) -> None:
    artifact = wandb.Artifact(name, type="model", metadata=metadata)
    artifact.add_file(str(path))
    run.log_artifact(artifact, aliases=["best"])
