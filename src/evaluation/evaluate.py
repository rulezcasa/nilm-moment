"""Evaluate a finished run's best checkpoint on a held-out split and log it to W&B.

Examples:
    python -m src.evaluation.evaluate --run-dir runs/cnn/kettle/final
    python -m src.evaluation.evaluate --run-dir runs/cnn/kettle/final runs/cnn/fridge_freezer/final --split val
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm

from src.data import NILMWindowDataset, Normalizer, make_window_loader
from src.models.cnn import count_parameters
from src.training.checkpoint import BEST
from src.training.tracking import TrackingInfo, start_run
from src.training.trainer import MODELS, load_split
from src.utils.seed import seed_everything

from .metrics import compute_metrics


def predict(model, dataset: NILMWindowDataset, config: Dict[str, Any], device) -> np.ndarray:
    """Normalized predictions for every window of ``dataset``, in dataset order."""

    loader, _ = make_window_loader(
        dataset,
        batch_size=config["training"]["batch_size"],
        max_per_house=max(dataset.windows_per_house),  # no subsampling
        shuffle=False,
        seed=config["seed"],
        num_workers=config["training"]["num_workers"],
    )
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, _ in tqdm(loader, leave=False):
            outputs.append(model(x.to(device)).squeeze(-1).cpu().numpy())
    return np.concatenate(outputs)


def evaluate_run(run_dir: Path, split: str = "test") -> Dict[str, Any]:
    """Score ``run_dir/best.pt`` on ``split``, write ``eval_<split>.json`` and log a W&B eval run."""

    checkpoint = torch.load(run_dir / BEST, map_location="cpu")
    config = checkpoint["config"]
    normalizer = Normalizer(**checkpoint["normalizer"])

    seed_everything(config["seed"])
    device = torch.device(config["device"])
    model = MODELS[config["model"]["name"]](config["model"], config["data"]["input_length"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    dataset = load_split(config, split)
    dataset.normalizer = normalizer
    y_pred = np.clip(normalizer.denormalize_target(predict(model, dataset, config, device)), 0, None).astype(np.float64)
    y_true = dataset.anchor_values()[1].astype(np.float64)

    bounds = np.cumsum(dataset.windows_per_house)[:-1]
    houses = [
        {"house": name, "windows": count, **compute_metrics(true, pred)}
        for name, count, true, pred in zip(
            dataset.house_names, dataset.windows_per_house, np.split(y_true, bounds), np.split(y_pred, bounds)
        )
    ]
    result = {"split": split, **compute_metrics(y_true, y_pred), "houses": houses}
    (run_dir / f"eval_{split}.json").write_text(json.dumps(result, indent=2))
    print(f"{run_dir} {split}: mae={result['mae']:.3f} W sae={result['sae']:.4f}")

    tracking = TrackingInfo(
        name=f"{'/'.join(run_dir.parts[-3:])}/eval-{split}",
        group="/".join(run_dir.parts[-3:-1]),
        job_type="eval",
        meta={
            "run_dir": str(run_dir),
            "split": split,
            "device": str(device),
            "parameters": count_parameters(model),
            "checkpoint": {"epoch": checkpoint["epoch"], "val_loss": checkpoint["val_loss"]},
            "normalizer": normalizer.to_dict(),
            "data": {f"{split}_houses": dict(zip(dataset.house_names, dataset.windows_per_house))},
        },
        tables={f"{split}/per_house": houses},
    )
    run = start_run(config, run_dir, tracking)
    run.summary.update({f"{split}/mae": result["mae"], f"{split}/sae": result["sae"]})
    run.finish()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, nargs="+", type=Path, help="run directories containing best.pt")
    parser.add_argument("--split", default="test", choices=("test", "val"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for run_dir in args.run_dir:
        evaluate_run(run_dir, args.split)


if __name__ == "__main__":
    main()
