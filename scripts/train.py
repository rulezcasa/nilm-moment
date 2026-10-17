"""Train the paper-style CNN sequence-to-point NILM baseline.

Example:
    python scripts/train.py --target Kettle --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.cnn import CNNConfig, CNNSeq2Point, count_parameters
from src.training.dataset import NILMWindowDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=["Kettle", "Fridge-Freezer", "Washing Machine"])
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cnn"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-windows-per-file", type=int)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(files, target, data_cfg, training_cfg, shuffle):
    dataset = NILMWindowDataset(
        files=files,
        target=target,
        input_length=data_cfg["input_length"],
        input_mean=data_cfg["input_mean"],
        input_std=data_cfg["input_std"],
        target_mean=data_cfg["target_mean"],
        target_std=data_cfg["target_std"],
        max_windows_per_file=training_cfg["max_windows_per_file"],
        shuffle=shuffle,
        drop_target_above_aggregate=data_cfg["drop_target_above_aggregate"],
        seed=training_cfg["seed"],
    )
    return DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None):
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


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    data_cfg = config["data"]
    training_cfg = config["training"]
    training_cfg["seed"] = config.get("seed", 42)
    if args.data_root:
        data_cfg["root"] = str(args.data_root)
    if args.max_windows_per_file is not None:
        training_cfg["max_windows_per_file"] = args.max_windows_per_file
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs

    seed_everything(training_cfg["seed"])
    device_name = config.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
    if device_name != "auto":
        device = torch.device(device_name)

    root = Path(data_cfg["root"]) / args.target
    train_files = sorted((root / "train").glob("*.csv"))
    val_files = sorted((root / "val").glob("*.csv"))
    if not train_files or not val_files:
        raise FileNotFoundError(f"expected train and val CSVs below {root}")

    model_cfg = config["model"]
    model = CNNSeq2Point(CNNConfig(
        input_length=data_cfg["input_length"],
        conv_channels=tuple(model_cfg["conv_channels"]),
        kernel_sizes=tuple(model_cfg["kernel_sizes"]),
        strides=tuple(model_cfg["strides"]),
        dense_units=model_cfg["dense_units"],
        dropout=model_cfg["dropout"],
    )).to(device)
    print(f"device={device} parameters={count_parameters(model):,}")

    train_loader = make_loader(train_files, args.target, data_cfg, training_cfg, shuffle=True)
    val_loader = make_loader(val_files, args.target, data_cfg, training_cfg, shuffle=False)
    loss_fn = nn.SmoothL1Loss() if training_cfg["loss"] == "smooth_l1" else nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(1, training_cfg["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, loss_fn, device)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "config": config, "target": args.target}, args.output_dir / f"{args.target}.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= training_cfg["patience"]:
                print("early stopping")
                break


if __name__ == "__main__":
    main()
