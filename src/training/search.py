"""Random hyperparameter search driven by a model config's ``search`` block.

Each trial trains one sampled parameter combination on the reduced
``search.budget`` and is ranked by its best ``search.metric``; the best
trial's parameters are then retrained on the full budget. Layout::

    runs/<model>/<appliance>/search-<name>/
        config.yaml   merged config of the search
        trials.yaml   the sampled combinations; trial i trains entry i
        results.csv   one row per finished trial
        trial-00/ ... ordinary run directories (see checkpoint.py)
        final/        best parameters, full budget

A search can be stopped at any time and continued by running it again with
the same name. A run directory with ``result.json`` is finished and skipped;
one without it was interrupted and is deleted and retrained from epoch 1.
``trials.yaml`` is written once and read back, so trial i is always the same
combination; raising ``search.trials`` appends new combinations after
checking that re-sampling reproduces the saved ones.
"""

import csv
import math
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import yaml

from src.data import Normalizer

from .checkpoint import CONFIG, create_run_dir, read_config, save_config
from .tracking import TrackingInfo
from .trainer import RunResult, finished_run, load_split, train_run

TRIALS = "trials.yaml"

# Search keys may only touch these sections: data settings change the
# datasets, which are loaded once and shared by every trial.
TUNABLE_SECTIONS = ("model", "training")
METRICS = ("val_loss",)


def _section_of(config: Dict[str, Any], key: str) -> str:
    sections = [name for name in TUNABLE_SECTIONS if key in config.get(name, {})]
    if len(sections) != 1:
        found = f"found in {sections}" if sections else "not found"
        raise ValueError(f"search key {key!r} must name exactly one setting in {TUNABLE_SECTIONS}; {found}")
    return sections[0]


def apply_params(config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``config`` with each key set in the section that defines it."""

    config = deepcopy(config)
    for key, value in params.items():
        config[_section_of(config, key)][key] = value
    return config


def _sample(spec: Any, rng: np.random.Generator) -> Any:
    if isinstance(spec, list):
        return spec[int(rng.integers(len(spec)))]
    if isinstance(spec, dict) and len(spec) == 1:
        (kind, bounds), = spec.items()
        low, high = bounds
        if kind == "log_uniform":
            return float(math.exp(rng.uniform(math.log(low), math.log(high))))
        if kind == "uniform":
            return float(rng.uniform(low, high))
    raise ValueError(f"unsupported search space entry {spec!r}; use a list, {{uniform: [a, b]}} or {{log_uniform: [a, b]}}")


def sample_trials(space: Dict[str, Any], trials: int, seed: int) -> Iterator[Dict[str, Any]]:
    """Yield ``trials`` parameter sets; the same seed always yields the same sequence."""

    rng = np.random.default_rng(seed)
    for _ in range(trials):
        yield {key: _sample(spec, rng) for key, spec in space.items()}


def _write_results(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _without_trial_count(config: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(config)
    config["search"].pop("trials", None)
    return config


def _open_search_dir(config: Dict[str, Any], search_dir: Path) -> List[Dict[str, Any]]:
    """Create or reopen ``search_dir`` and return its trial combinations."""

    search_cfg = config["search"]
    sampled = list(sample_trials(search_cfg["space"], search_cfg["trials"], config["seed"]))
    if not search_dir.exists():
        create_run_dir(search_dir)
        save_config(search_dir, config)
        (search_dir / TRIALS).write_text(yaml.safe_dump(sampled, sort_keys=False))
        return sampled

    if _without_trial_count(read_config(search_dir)) != _without_trial_count(config):
        raise ValueError(f"{search_dir} was started with a different config; only the number of trials may change")
    saved = yaml.safe_load((search_dir / TRIALS).read_text()) or []
    if len(sampled) > len(saved):
        if sampled[: len(saved)] != saved:
            raise ValueError(f"re-sampling no longer reproduces {search_dir / TRIALS}; cannot extend this search")
        saved = sampled
        (search_dir / TRIALS).write_text(yaml.safe_dump(saved, sort_keys=False))
        save_config(search_dir, config)
        print(f"extended {search_dir.name} to {len(saved)} trials")
    elif len(sampled) < len(saved):
        print(f"{search_dir.name} has {len(saved)} trials; running the first {len(sampled)}")
        saved = saved[: len(sampled)]
    print(f"continuing {search_dir}")
    return saved


def _trained_with_other_settings(run_dir: Path, expected: Dict[str, Any]) -> bool:
    """Whether ``run_dir`` holds a run whose model or training settings differ from ``expected``."""

    if not (run_dir / CONFIG).exists():
        return False
    saved = read_config(run_dir)
    return any(saved[section] != expected[section] for section in TUNABLE_SECTIONS)


def _reuse_or_clear(run_dir: Path) -> Optional[RunResult]:
    """Return the finished result in ``run_dir``; otherwise delete any partial run so it can be retrained."""

    if not run_dir.exists():
        return None
    result = finished_run(run_dir)
    if result is not None:
        return result
    print(f"discarding unfinished {run_dir.name}")
    shutil.rmtree(run_dir)
    return None


def run_search(config: Dict[str, Any], search_dir: Path, retrain: bool = True) -> RunResult:
    """Run or continue the search in ``search_dir``; return the final run, or the best trial if ``retrain`` is off."""

    search_cfg = config.get("search")
    if not search_cfg:
        raise ValueError(f"model {config['model']['name']!r} has no search block")
    if search_cfg["method"] != "random":
        raise ValueError(f"unsupported search method {search_cfg['method']!r}")
    if search_cfg["metric"] not in METRICS:
        raise ValueError(f"unsupported search metric {search_cfg['metric']!r}; expected one of {METRICS}")

    budget_config = apply_params(config, search_cfg.get("budget", {}))
    trials = _open_search_dir(config, search_dir)
    for params in trials:
        apply_params(budget_config, params)  # fail on a bad key before any training

    # Load lazily and once: every trial and the final run share the data and the training-split statistics.
    shared: Optional[Tuple[Any, Any, Normalizer]] = None

    def data() -> Tuple[Any, Any, Normalizer]:
        nonlocal shared
        if shared is None:
            train_set = load_split(config, "train")
            val_set = load_split(config, "val")
            shared = (train_set, val_set, Normalizer.fit(*train_set.anchor_values()))
        return shared

    # W&B: trials and the final run share one group per search, e.g. cnn/kettle/search-v1.
    group = "/".join(search_dir.parts[-3:])
    search_meta = {"name": search_dir.name, "trials": len(trials), "space": search_cfg["space"], "budget": search_cfg.get("budget", {})}

    rows = []
    best_index = None
    best_result = None
    for index, params in enumerate(trials):
        trial_dir = search_dir / f"trial-{index:02d}"
        trial_config = apply_params(budget_config, params)
        if _trained_with_other_settings(trial_dir, trial_config):
            raise ValueError(f"{trial_dir} was trained with different settings than entry {index} of {TRIALS}")
        result = _reuse_or_clear(trial_dir)
        if result is not None:
            print(f"trial {index + 1}/{len(trials)} already done: {params} val_loss={result.best_val:.6f}")
        else:
            print(f"\n=== trial {index + 1}/{len(trials)}: {params}")
            tracking = TrackingInfo(
                name=f"{group}/{trial_dir.name}",
                group=group,
                job_type="search-trial",
                meta={"search": {**search_meta, "trial": index, "params": params}},
            )
            result = train_run(trial_config, trial_dir, *data(), tracking)
        rows.append({"trial": index, **params, "best_val_loss": result.best_val, "best_epoch": result.best_epoch, "epochs": result.last_epoch})
        _write_results(search_dir / "results.csv", rows)
        if best_result is None or result.best_val < best_result.best_val:
            best_index, best_result = index, result

    best_params = trials[best_index]
    print(f"\nbest trial {best_index}: {best_params} val_loss={best_result.best_val:.6f}")
    if not retrain:
        return best_result

    final_dir = search_dir / "final"
    final_config = apply_params(config, best_params)
    if _trained_with_other_settings(final_dir, final_config):
        # More trials found a better combination than the one final/ was trained with.
        print(f"best trial changed since {final_dir.name} was trained; retraining it")
        shutil.rmtree(final_dir)
    final = _reuse_or_clear(final_dir)
    if final is not None:
        print(f"final already done: val_loss={final.best_val:.6f}")
        return final
    print("=== retraining best parameters on the full budget")
    tracking = TrackingInfo(
        name=f"{group}/final",
        group=group,
        job_type="search-final",
        meta={"search": {**search_meta, "best_trial": best_index, "params": best_params, "best_trial_val_loss": best_result.best_val}},
        tables={"search_results": rows},
    )
    return train_run(final_config, final_dir, *data(), tracking)
