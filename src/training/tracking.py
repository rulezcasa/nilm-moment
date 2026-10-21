"""Weights & Biases logging: one W&B run per run directory.

Runs are organised so the W&B UI splits them by appliance and by
hyperparameters:

    project   wandb.project, one project for the whole study
    group     <model>/<appliance>                 standalone runs
              <model>/<appliance>/search-<name>   a search's trials and final run
    job_type  train | search-trial | search-final
    tags      model, appliance, job type
    name      run directory below runs/, e.g. cnn/kettle/search-v1/trial-03
    config    the merged run config, so every hyperparameter is a filterable
              key (model.dense_units, training.learning_rate, ...), plus
              ``meta``: data, normalization and search metadata

Set ``wandb.mode`` to ``offline`` to sync later, or ``disabled`` to turn
logging off without importing wandb.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGED_MODEL_JOBS = ("train", "search-final")


@dataclass(frozen=True)
class TrackingInfo:
    """How a run is labelled in W&B; built by the caller that knows the run's role."""

    name: str
    group: str
    job_type: str = "train"
    meta: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


class Tracker:
    """Thin wrapper so the trainer works the same with logging disabled."""

    def __init__(self, run=None, log_model: bool = False) -> None:
        self.run = run
        self.log_model = log_model

    @classmethod
    def start(cls, config: Dict[str, Any], run_dir: Path, info: TrackingInfo) -> "Tracker":
        settings = config.get("wandb") or {}
        mode = settings.get("mode", "online")
        if mode == "disabled":
            return cls()

        import wandb

        model, target = config["model"]["name"], config["target"]
        run = wandb.init(
            project=settings.get("project", "nilm-moment"),
            entity=settings.get("entity"),
            mode=mode,
            dir=str(run_dir),
            name=info.name,
            group=info.group,
            job_type=info.job_type,
            tags=[model, target, info.job_type],
            config={**config, "meta": info.meta},
        )
        run.define_metric("epoch")
        run.define_metric("*", step_metric="epoch")
        run.define_metric("val_loss", summary="min")
        for key, rows in info.tables.items():
            if rows:
                run.log({key: wandb.Table(columns=list(rows[0]), data=[list(row.values()) for row in rows])})
        return cls(run, log_model=bool(settings.get("log_model", False)) and info.job_type in LOGGED_MODEL_JOBS)

    def log_epoch(self, values: Dict[str, Any]) -> None:
        if self.run:
            self.run.log(values)

    def summarize(self, values: Dict[str, Any]) -> None:
        if self.run:
            self.run.summary.update(values)

    def upload_model(self, path: Path, name: str, metadata: Dict[str, Any]) -> None:
        if not (self.run and self.log_model):
            return
        import wandb

        artifact = wandb.Artifact(name, type="model", metadata=metadata)
        artifact.add_file(str(path))
        self.run.log_artifact(artifact, aliases=["best"])

    def finish(self, exit_code: Optional[int] = 0) -> None:
        if self.run:
            self.run.finish(exit_code=exit_code)
