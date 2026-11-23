"""Predict one contiguous test span per appliance with each model's best seed, for prediction-vs-actual plots.

Writes spans/<appliance>.npz (time, mains, truth, pred_<model>) and spans/<appliance>.json (house, seeds, span MAE).

Example:
    python -m scripts.predict_span --target Kettle
    python -m scripts.predict_span --target Kettle --house SEGMENTED_HOUSE2 --start 1379455691 --hours 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import Normalizer, split_files
from src.data.windowing import anchor_offset, window_starts
from src.training.checkpoint import BEST, RUNS_DIR
from src.training.trainer import MODELS
from src.utils.config import appliance_slug, load_config

OUT_DIR = Path("spans")
BATCH_SIZE = 256
ON_THRESHOLD = {"Kettle": 2000.0, "Fridge-Freezer": 50.0, "Washing Machine": 20.0}  # watts, as in README


def best_seed_dir(model: str, target: str) -> tuple[Path, float]:
    """Seed run directory with the lowest test MAE that still has a checkpoint."""

    scored = []
    for run_dir in (RUNS_DIR / model / appliance_slug(target)).iterdir():
        result = run_dir / "eval_test.json"
        if run_dir.name.isdigit() and result.exists() and (run_dir / BEST).exists():
            scored.append((json.loads(result.read_text())["mae"], run_dir))
    if not scored:
        raise FileNotFoundError(f"no evaluated checkpoint for {model}/{target}")
    mae, run_dir = min(scored)
    return run_dir, mae


def load_house_frame(path: Path, target: str) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["Unix", "Aggregate", target, "Segment"])


def window_sums(values: np.ndarray, length: int) -> np.ndarray:
    """Sum of every ``length``-long run of ``values``."""

    total = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return total[length:] - total[:-length]


def span_starts(
    frame: pd.DataFrame, target: str, input_length: int, length: int, start_unix: int | None
) -> tuple[np.ndarray, float]:
    """Window starts of ``length`` consecutive anchors in one segment and their activation count.

    At ``start_unix``, or the span with the most activations among spans whose appliance
    reading never exceeds mains (metering artefacts); target energy breaks ties.
    """

    anchor = anchor_offset(input_length)
    starts = window_starts(frame["Segment"].to_numpy(), input_length)
    anchors = starts + anchor
    if len(anchors) < length:
        raise ValueError("house shorter than the requested span")
    contiguous = anchors[length - 1 :] - anchors[: len(anchors) - length + 1] == length - 1

    if start_unix is not None:
        i = int(np.searchsorted(frame["Unix"].to_numpy()[anchors], start_unix))
        if i >= len(contiguous) or not contiguous[i]:
            raise ValueError(f"span at {start_unix} is not inside one segment")
        return starts[i : i + length], float("nan")

    target_w = frame[target].to_numpy(np.float64)[anchors]
    artefact = target_w > frame["Aggregate"].to_numpy(np.float64)[anchors]
    on = target_w >= ON_THRESHOLD[target]
    rising = np.concatenate(([False], on[1:] & ~on[:-1]))

    activations = window_sums(rising, length)
    score = activations + window_sums(target_w, length) / (length * 1e6)  # mean watts / 1e6 < 1: tie-break only
    score[~contiguous | (window_sums(artefact, length) > 0)] = -np.inf
    i = int(np.argmax(score))
    if not np.isfinite(score[i]):
        raise ValueError("no artefact-free span inside one segment")
    return starts[i : i + length], float(activations[i])


def predict_windows(run_dir: Path, aggregate: np.ndarray, starts: np.ndarray, device: torch.device) -> np.ndarray:
    """Watts predicted by ``run_dir/best.pt`` at each window's anchor."""

    checkpoint = torch.load(run_dir / BEST, map_location="cpu")
    config = checkpoint["config"]
    normalizer = Normalizer(**checkpoint["normalizer"])
    input_length = config["data"]["input_length"]
    model = MODELS[config["model"]["name"]](config["model"], input_length)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    window = np.arange(input_length)
    outputs = []
    with torch.no_grad():
        for batch in np.array_split(starts, max(1, len(starts) // BATCH_SIZE)):
            x = normalizer.normalize_input(aggregate[batch[:, None] + window]).astype(np.float32)
            outputs.append(model(torch.from_numpy(x).to(device)).squeeze(-1).cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.clip(normalizer.denormalize_target(np.concatenate(outputs)), 0, None).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one test span per appliance for plotting.")
    parser.add_argument("--target", required=True, help="appliance column, e.g. Kettle")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--house", help="test house CSV stem; default: house with the highest-energy span")
    parser.add_argument("--start", type=int, help="span start, Unix seconds (needs --house)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.start is not None and args.house is None:
        parser.error("--start needs --house")

    config = load_config(args.models[0], args.target)
    input_length = config["data"]["input_length"]
    anchor = anchor_offset(input_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = split_files(config["data"]["root"], args.target, "test")
    if args.house:
        files = [f for f in files if f.stem == args.house]
        if not files:
            raise FileNotFoundError(f"test house {args.house} not found")

    best = None
    for path in files:
        frame = load_house_frame(path, args.target)
        length = int(round(args.hours * 3600 / np.median(np.diff(frame["Unix"].to_numpy()))))
        try:
            starts, activations = span_starts(frame, args.target, input_length, length, args.start)
        except ValueError as error:
            print(f"  skip {path.stem}: {error}")
            continue
        print(f"  {path.stem}: {activations:.0f} activations")
        if best is None or activations > best[0]:
            best = (activations, path.stem, frame, starts)
    if best is None:
        raise ValueError(f"no usable span for {args.target}")
    activations, house, frame, starts = best
    anchors = starts + anchor
    aggregate = frame["Aggregate"].to_numpy(np.float32)
    truth = frame[args.target].to_numpy(np.float32)[anchors]
    print(f"{args.target}: {house}, {len(starts):,} samples from Unix {int(frame['Unix'].iloc[anchors[0]])}")

    arrays = {
        "unix": frame["Unix"].to_numpy(np.int64)[anchors],
        "aggregate": aggregate[anchors],
        "truth": truth,
    }
    meta = {
        "target": args.target,
        "house": house,
        "hours": args.hours,
        "samples": len(starts),
        "activations": activations,
        "models": {},
    }
    for model in args.models:
        run_dir, test_mae = best_seed_dir(model, args.target)
        pred = predict_windows(run_dir, aggregate, starts, device)
        arrays[f"pred_{model}"] = pred
        span_mae = float(np.abs(pred - truth).mean())
        meta["models"][model] = {"seed": int(run_dir.name), "test_mae": test_mae, "span_mae": span_mae}
        print(f"  {model:16s} seed {run_dir.name}  test MAE {test_mae:.2f}  span MAE {span_mae:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    slug = appliance_slug(args.target)
    np.savez_compressed(args.out / f"{slug}.npz", **arrays)
    (args.out / f"{slug}.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out / slug}.npz/.json")


if __name__ == "__main__":
    main()
