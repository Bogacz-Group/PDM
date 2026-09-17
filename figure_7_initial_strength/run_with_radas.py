#!/usr/bin/env python3
"""Run a fresh Figure 7b search with the public mini-radas package."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if __package__:
    from . import fit_initial_strength
else:
    sys.path.insert(0, str(HERE))
    import fit_initial_strength

MINI_RADAS_COMMIT = "8314d91b773db5148b0a90a8442f549e367e9cfb"
MINI_RADAS_SOURCE_SHA256 = (
    "b0b85dcea68d1992c1cfd5e39725d41e93e8826ed55c332ad20870672191dda4"
)
MINI_RADAS_SOURCE_FILES = ("__init__.py", "core.py", "run_experiment.py", "utils.py")
SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
DEFAULT_STORAGE = HERE / "generated" / "radas_storage"
DEFAULT_OUTPUT_ROOT = HERE / "generated" / "radas"


def resolve_path(path: Path | str) -> Path:
    """Resolve relative paths from this experiment directory."""

    candidate = Path(path).expanduser()
    return (
        (HERE / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )


def validate_component(value: str, option: str) -> str:
    """Accept only one conservative path component for mini-radas names."""

    if value in {".", ".."} or SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{option} contains unsupported characters")
    return value


def _source_fingerprint(module: Any) -> str:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError("Cannot locate the imported radas package")
    package_dir = Path(module_file).resolve().parent
    digest = hashlib.sha256()
    for filename in MINI_RADAS_SOURCE_FILES:
        source = package_dir / filename
        if not source.is_file():
            raise RuntimeError(f"Imported radas package is missing {filename}")
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_mini_radas(module: Any, run_experiment: Callable[..., Any]) -> str:
    """Fail unless the imported package matches the pinned public source."""

    required = {"user_name", "resources", "run_with", "local_storage_path", "dos"}
    missing = sorted(required.difference(inspect.signature(run_experiment).parameters))
    if missing:
        raise RuntimeError(f"Imported radas API is missing parameters: {missing}")
    observed = _source_fingerprint(module)
    if observed != MINI_RADAS_SOURCE_SHA256:
        raise RuntimeError(
            "Imported radas sources do not match public mini-radas commit "
            f"{MINI_RADAS_COMMIT}: observed {observed}"
        )
    return observed


def sample_count(args: argparse.Namespace) -> tuple[str, int]:
    profile = "smoke" if args.smoke else "quick" if args.quick else "full"
    defaults = {"full": 1000, "quick": 32, "smoke": 1}
    count = args.num_samples if args.num_samples is not None else defaults[profile]
    if count < 1:
        raise ValueError("--num-samples must be at least 1")
    return profile, int(count)


def run_identifier(args: argparse.Namespace, profile: str, count: int) -> str:
    payload = {
        "profile": profile,
        "num_samples": count,
        "seed": args.seed,
        "search_space": args.search_space,
    }
    suffix = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    return f"{profile}-seed-{args.seed}-{suffix}"


def build_search(args: argparse.Namespace, count: int) -> tuple[dict[str, Any], Any]:
    """Build the Ray Tune parameter space and seeded Optuna search."""

    try:
        from ray import tune
        from ray.tune.search.optuna import OptunaSearch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Ray Tune and Optuna are required") from exc
    space = {
        name: (tune.loguniform(low, high) if logarithmic else tune.uniform(low, high))
        for name, (low, high, logarithmic) in fit_initial_strength.search_bounds(
            args.search_space
        ).items()
    }
    search = OptunaSearch(metric="objective", mode="min", seed=args.seed)
    config = tune.TuneConfig(
        metric="objective",
        mode="min",
        search_alg=search,
        num_samples=count,
        max_concurrent_trials=1,
    )
    return space, config


def _value(row: Mapping[str, Any], name: str) -> Any:
    if name in row and row[name] is not None:
        return row[name]
    config_name = f"config/{name}"
    if config_name in row and row[config_name] is not None:
        return row[config_name]
    raise KeyError(f"Ray results contain neither {name!r} nor {config_name!r}")


def fit_trials_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract stable fit columns from Ray's analysis table."""

    rows = []
    for trial, (_, series) in enumerate(frame.iterrows()):
        raw = series.to_dict()
        row = {
            "trial": trial,
            "objective": _value(raw, "objective"),
            **{
                name: _value(raw, name) for name in fit_initial_strength.PARAMETER_NAMES
            },
        }
        if "rmse" in raw:
            row["rmse"] = raw["rmse"]
        rows.append(row)
    trials = pd.DataFrame(rows)
    trials["objective"] = pd.to_numeric(trials["objective"], errors="coerce")
    return trials.sort_values("trial", kind="stable").reset_index(drop=True)


def materialize(
    args: argparse.Namespace,
    count: int,
    frame: pd.DataFrame,
    output: Path,
) -> dict[str, Path]:
    trials = fit_trials_frame(frame)
    finite = trials[np.isfinite(trials["objective"].to_numpy(dtype=float))]
    if finite.empty:
        raise RuntimeError("all mini-radas trials returned non-finite objectives")
    best = finite.loc[finite["objective"].idxmin()]
    params = {name: float(best[name]) for name in fit_initial_strength.PARAMETER_NAMES}
    paths = fit_initial_strength.materialize_result(
        data=fit_initial_strength.load_dataset(),
        params=params,
        output=output,
        formats=[] if args.no_plots else ["pdf", "svg"],
        dpi=args.dpi,
        source="mini-radas_optuna",
        search_space=args.search_space,
        seed=args.seed,
        num_samples=count,
    )
    trials_csv = output / "fig7b_initial_strength_trials.csv"
    trials.to_csv(trials_csv, index=False)
    paths["trials_csv"] = trials_csv
    return paths


async def run(args: argparse.Namespace) -> dict[str, Path]:
    """Execute the local public mini-radas search and write final outputs."""

    profile, count = sample_count(args)
    if args.seed < 0 or args.dpi <= 0 or args.cpus_per_trial <= 0:
        raise ValueError(
            "seed must be non-negative; dpi and CPU count must be positive"
        )
    run_id = run_identifier(args, profile, count)
    experiment = args.experiment_name or f"pdm-fig7b-{run_id}"
    validate_component(experiment, "--experiment-name")
    validate_component(args.user_name, "--user-name")

    try:
        import radas as radas_module
        from radas import run_experiment
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Install requirements.txt, including public mini-radas"
        ) from exc
    source_hash = validate_mini_radas(radas_module, run_experiment)
    parameter_space, tune_config = build_search(args, count)
    storage = resolve_path(args.storage)
    output = (
        resolve_path(args.output)
        if args.output is not None
        else (DEFAULT_OUTPUT_ROOT / run_id).resolve()
    )
    storage.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    previous_directory = Path.cwd()
    try:
        os.chdir(HERE)
        result = await run_experiment(
            user_name=args.user_name,
            trainable=fit_initial_strength.initial_strength_fit_trainable,
            experiment_name=experiment,
            resources={"cpu": float(args.cpus_per_trial), "gpu": 0},
            run_with="local",
            local_storage_path=str(storage),
            tuner_init_kwargs={
                "tune_config": tune_config,
                "run_config_kwargs": {"verbose": 0 if profile == "smoke" else 1},
            },
            param_space=parameter_space,
            dos=["run", "analyze"],
        )
    finally:
        os.chdir(previous_directory)

    frame = result.get("df")
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError("mini-radas did not return an analysis dataframe")
    if len(frame) != count:
        raise RuntimeError(f"mini-radas returned {len(frame)} trials; expected {count}")
    paths = materialize(args, count, frame, output)
    metadata = output / "fig7b_run.json"
    metadata.write_text(
        json.dumps(
            {
                "panel": "Figure 7b",
                "profile": profile,
                "run_id": run_id,
                "seed": args.seed,
                "requested_trials": count,
                "completed_trials": len(frame),
                "search_space": args.search_space,
                "orchestrator": {
                    "package": "mini-radas",
                    "expected_commit": MINI_RADAS_COMMIT,
                    "verified_source_sha256": source_hash,
                    "api": "radas.run_experiment",
                },
                "files": {name: str(path) for name, path in sorted(paths.items())},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["run_metadata"] = metadata
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--quick", action="store_true", help="run 32 trials")
    profile.add_argument("--smoke", action="store_true", help="run one trial")
    parser.add_argument(
        "--num-samples", type=int, help="override the profile trial count"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--search-space", choices=("legacy", "manuscript"), default="legacy"
    )
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--user-name", default="pdm-public")
    parser.add_argument("--experiment-name")
    parser.add_argument("--cpus-per-trial", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = asyncio.run(run(args))
    print("Completed Figure 7b search with public mini-radas.")
    for label, path in sorted(paths.items()):
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
