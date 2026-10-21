"""Train one config into one run directory.

Shared by scripts/train.py (single runs) and src/training/search.py (search
trials and the final retrain), so datasets can be loaded once and reused.
"""

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from tqdm import tqdm

from src.data import NILMWindowDataset, Normalizer, make_window_loader, split_files
from src.models.cnn import build_cnn, count_parameters
from src.utils.config import appliance_slug
from src.utils.seed import seed_everything

from .checkpoint import BEST, append_metrics, create_run_dir, read_result, save_best, save_config, save_result
from .tracking import Tracker, TrackingInfo

LOSSES = {"mse": nn.MSELoss, "smooth_l1": nn.SmoothL1Loss}
OPTIMIZERS = {"adam": torch.optim.Adam}


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    best_val: float
    best_epoch: int
    last_epoch: int


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(config: Dict[str, Any]) -> nn.Module:
    name = config["model"]["name"]
    if name == "cnn":
        return build_cnn(config["model"], config["data"]["input_length"])
    raise NotImplementedError(f"training for model {name!r} is not implemented yet")


def _lookup(table: Dict[str, Any], key: str, kind: str):
    if key not in table:
        raise ValueError(f"unknown {kind} {key!r}; expected one of {sorted(table)}")
    return table[key]


def load_split(config: Dict[str, Any], split: str) -> NILMWindowDataset:
    data_cfg = config["data"]
    dataset = NILMWindowDataset.from_files(
        split_files(data_cfg["root"], config["target"], split),
        config["target"],
        input_length=data_cfg["input_length"],
        drop_target_above_aggregate=data_cfg["drop_target_above_aggregate"],
    )
    print(f"{split}: {dataset.summary()}")
    return dataset


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(x).squeeze(-1)
        loss = loss_fn(prediction, y)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        batches += 1
    if batches == 0:
        raise RuntimeError("the loader produced no windows")
    return total_loss / batches


def train_run(
    config: Dict[str, Any],
    run_dir: Path,
    train_set: NILMWindowDataset,
    val_set: NILMWindowDataset,
    normalizer: Normalizer,
    tracking: Optional[TrackingInfo] = None,
) -> RunResult:
    """Train ``config`` into the new directory ``run_dir`` and log it to W&B.

    ``normalizer`` must be fitted on ``train_set``; it is applied to both
    datasets and saved with the best checkpoint. ``tracking`` labels the W&B
    run; by default it is a standalone run named after ``run_dir``.
    """

    training_cfg = config["training"]
    loss_fn = _lookup(LOSSES, training_cfg["loss"], "loss")()
    optimizer_cls = _lookup(OPTIMIZERS, training_cfg["optimizer"], "optimizer")
    if tracking is None:
        tracking = TrackingInfo(name="/".join(run_dir.parts[-3:]), group="/".join(run_dir.parts[-3:-1]))

    create_run_dir(run_dir)
    save_config(run_dir, config)
    train_set.normalizer = val_set.normalizer = normalizer

    seed_everything(config["seed"])
    device = resolve_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    optimizer = optimizer_cls(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    pin_memory = device.type == "cuda"
    loader_args = dict(
        batch_size=training_cfg["batch_size"],
        max_per_house=training_cfg["max_windows_per_file"],
        seed=config["seed"],
        num_workers=training_cfg["num_workers"],
        pin_memory=pin_memory,
    )
    train_loader, train_sampler = make_window_loader(train_set, shuffle=True, **loader_args)
    val_loader, val_sampler = make_window_loader(val_set, shuffle=False, **loader_args)

    print(f"run directory: {run_dir}")
    print(f"device={device} parameters={count_parameters(model):,} normalization: {normalizer}")
    meta = {
        **tracking.meta,
        "run_dir": str(run_dir),
        "device": str(device),
        "parameters": count_parameters(model),
        "normalizer": normalizer.to_dict(),
        "data": {
            "train_houses": dict(zip(train_set.house_names, train_set.windows_per_house)),
            "val_houses": dict(zip(val_set.house_names, val_set.windows_per_house)),
            "train_windows_per_epoch": len(train_sampler),
            "val_windows": len(val_sampler),
        },
    }
    tracker = Tracker.start(config, run_dir, replace(tracking, meta=meta))

    best_val, best_epoch, stale_epochs = float("inf"), 0, 0
    epoch = 0
    try:
        for epoch in range(1, training_cfg["epochs"] + 1):
            started = time.time()
            train_sampler.set_epoch(epoch)
            train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
            with torch.no_grad():
                val_loss = run_epoch(model, val_loader, loss_fn, device)

            is_best = val_loss < best_val
            if is_best:
                best_val, best_epoch, stale_epochs = val_loss, epoch, 0
                save_best(run_dir, model=model, config=config, normalizer=normalizer, epoch=epoch, val_loss=val_loss)
            else:
                stale_epochs += 1
            seconds = time.time() - started
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "is_best": is_best, "seconds": round(seconds, 1)}
            append_metrics(run_dir, row)
            tracker.log_epoch({**row, "best_val_loss": best_val})
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
                f"{' *best' if is_best else ''} ({seconds:.0f}s)"
            )
            if stale_epochs >= training_cfg["patience"]:
                print("early stopping")
                break

        print(f"best epoch {best_epoch} val_loss={best_val:.6f} -> {run_dir / BEST}")
        summary = {
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "stopped_early": stale_epochs >= training_cfg["patience"],
        }
        tracker.summarize(summary)
        tracker.upload_model(
            run_dir / BEST,
            name=f"{config['model']['name']}-{appliance_slug(config['target'])}",
            metadata={"run": tracking.name, "best_epoch": best_epoch, "val_loss": best_val, "normalizer": normalizer.to_dict()},
        )
        save_result(run_dir, summary)  # last, so it only exists for finished runs
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    tracker.finish()
    return RunResult(run_dir, best_val, best_epoch, epoch)


def finished_run(run_dir: Path) -> Optional[RunResult]:
    """The result of a run that completed, or None if it never started or was interrupted."""

    result = read_result(run_dir)
    if result is None:
        return None
    return RunResult(run_dir, result["best_val_loss"], result["best_epoch"], result["epochs_run"])
