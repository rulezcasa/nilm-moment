"""Grid hyperparameter search over a model config's search block."""

import csv
import itertools
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data import Normalizer

from .checkpoint import create_run_dir, read_config, save_config
from .tracking import TrackingInfo
from .trainer import RunResult, finished_run, load_split, train_run


def apply_params(config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``config`` without its search block, with each key set in the model or training section."""

    config = deepcopy(config)
    config.pop("search", None)
    for key, value in params.items():
        config["model" if key in config["model"] else "training"][key] = value
    return config


def grid_trials(space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Every combination of the space's lists; the first key varies slowest."""

    return [dict(zip(space, values)) for values in itertools.product(*space.values())]


def _settings(config: Dict[str, Any]) -> tuple:
    return config["data"], config["model"], config["training"], config["search"]["budget"], config["search"]["space"]


def _open_search_dir(config: Dict[str, Any], search_dir: Path) -> None:
    if not search_dir.exists():
        create_run_dir(search_dir)
        save_config(search_dir, config)
    elif _settings(read_config(search_dir)) != _settings(config):
        raise ValueError(f"{search_dir} was started with a different config")
    else:
        print(f"continuing {search_dir}")


def _write_results(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _reuse_or_clear(run_dir: Path) -> Optional[RunResult]:
    """Return the finished result in ``run_dir``; otherwise delete any partial run so it can be retrained."""

    result = finished_run(run_dir)
    if result is None and run_dir.exists():
        print(f"discarding unfinished {run_dir.name}")
        shutil.rmtree(run_dir)
    return result


def run_search(config: Dict[str, Any], search_dir: Path, retrain: bool = False) -> RunResult:
    """Run or continue the search in ``search_dir``; return the best trial, or the final run if ``retrain`` is on."""

    search_cfg = config["search"]
    budget_config = apply_params(config, search_cfg["budget"])
    trials = grid_trials(search_cfg["space"])
    _open_search_dir(config, search_dir)

    train_set = load_split(config, "train")
    val_set = load_split(config, "val")
    normalizer = Normalizer.fit(*train_set.anchor_values())

    group = "/".join(search_dir.parts[-3:])

    rows = []
    best_index = None
    best_result = None
    for index, params in enumerate(trials):
        trial_dir = search_dir / f"trial-{index:02d}"
        result = _reuse_or_clear(trial_dir)
        if result is not None:
            print(f"trial {index + 1}/{len(trials)} already done: {params} val_loss={result.best_val:.6f}")
        else:
            print(f"\n=== trial {index + 1}/{len(trials)}: {params}")
            tracking = TrackingInfo(
                name=f"{group}/{trial_dir.name}",
                group=group,
                job_type="search-trial",
                meta={"search": {"name": search_dir.name, "trial": index, "params": params}},
            )
            result = train_run(apply_params(budget_config, params), trial_dir, train_set, val_set, normalizer, tracking)
        rows.append({"trial": index, **params, "best_val_loss": result.best_val, "best_epoch": result.best_epoch, "epochs": result.last_epoch})
        _write_results(search_dir / "results.csv", rows)
        if best_result is None or result.best_val < best_result.best_val:
            best_index, best_result = index, result

    best_params = trials[best_index]
    print(f"\nbest trial {best_index}: {best_params} val_loss={best_result.best_val:.6f}")
    if not retrain:
        return best_result

    final_dir = search_dir / "final"
    final = _reuse_or_clear(final_dir)
    if final is not None:
        print(f"final already done: val_loss={final.best_val:.6f}")
        return final
    print("=== retraining best parameters on the full budget")
    tracking = TrackingInfo(
        name=f"{group}/final",
        group=group,
        job_type="search-final",
        meta={"search": {"name": search_dir.name, "best_trial": best_index, "params": best_params, "best_trial_val_loss": best_result.best_val}},
        tables={"search_results": rows},
    )
    return train_run(apply_params(config, best_params), final_dir, train_set, val_set, normalizer, tracking)
