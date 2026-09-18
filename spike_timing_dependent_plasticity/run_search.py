#!/usr/bin/env python3
"""Run fresh Figure 6i/6j searches with the pinned public mini-radas API."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fit_stdp  # noqa: E402


MINI_RADAS_COMMIT = "8314d91b773db5148b0a90a8442f549e367e9cfb"
MINI_RADAS_SOURCE_SHA256 = (
    "b0b85dcea68d1992c1cfd5e39725d41e93e8826ed55c332ad20870672191dda4"
)
MINI_RADAS_SOURCE_FILES = ("__init__.py", "core.py", "run_experiment.py", "utils.py")
TARGETS = ("fig6i", "fig6j")
DEFAULT_STORAGE = HERE / "generated" / "radas_storage"
DEFAULT_OUTPUT_ROOT = HERE / "generated" / "radas"
DEFAULT_USER_NAME = "pdm-public"
SAFE_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True)
class SearchProfile:
    """Resolved search size and its descriptive name."""

    name: str
    num_samples: int


def resolve_path(path: Path | str) -> Path:
    """Resolve relative paths from this experiment directory."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = HERE / candidate
    return candidate.resolve()


def validate_storage_component(value: str, option: str) -> str:
    """Reject traversal and separators in mini-radas directory components."""

    if value in {".", ".."} or SAFE_COMPONENT_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{option} must match {SAFE_COMPONENT_PATTERN.pattern!r} "
            "and cannot be '.' or '..'"
        )
    return value


def _mini_radas_source_fingerprint(module: Any) -> str:
    """Hash the source files expected at the pinned public revision."""

    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError("Cannot locate the imported radas package sources")
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
    """Fail closed unless the installed package is the pinned mini-radas source."""

    parameters = inspect.signature(run_experiment).parameters
    required = {"user_name", "resources", "run_with", "local_storage_path", "dos"}
    missing = sorted(required.difference(parameters))
    if missing:
        raise RuntimeError(
            "The imported radas API is not the pinned public mini-radas API; "
            f"missing parameters: {missing}"
        )
    observed = _mini_radas_source_fingerprint(module)
    if observed != MINI_RADAS_SOURCE_SHA256:
        raise RuntimeError(
            "The imported radas sources do not match mini-radas commit "
            f"{MINI_RADAS_COMMIT}. Expected source SHA-256 "
            f"{MINI_RADAS_SOURCE_SHA256}, observed {observed}."
        )
    return observed


def resolve_profile(args: argparse.Namespace) -> SearchProfile:
    if args.smoke:
        name, default_samples = "smoke", 1
    elif args.quick:
        name, default_samples = "quick", 32
    else:
        name, default_samples = "full", 1000
    num_samples = default_samples if args.num_samples is None else args.num_samples
    if num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    return SearchProfile(name, int(num_samples))


def run_identifier(args: argparse.Namespace, profile: SearchProfile) -> str:
    payload = {
        "target": args.target,
        "profile": profile.name,
        "num_samples": profile.num_samples,
        "seed": int(args.seed),
        "weighting": args.weighting,
        "variance_floor": float(args.variance_floor),
    }
    suffix = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return f"{profile.name}-seed-{args.seed}-{suffix}"


def build_search(args: argparse.Namespace, profile: SearchProfile) -> tuple[Any, Any]:
    """Build the Figure 6 fit trainable and Ray Tune search configuration."""

    try:
        from ray import tune
        from ray.tune.search.optuna import OptunaSearch
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise RuntimeError(
            "Ray Tune and Optuna are required; install requirements-search.txt."
        ) from exc

    search = OptunaSearch(metric="objective", mode="min", seed=args.seed)
    tune_config = tune.TuneConfig(
        metric="objective",
        mode="min",
        search_alg=search,
        num_samples=profile.num_samples,
        max_concurrent_trials=1,
    )
    dataset = "bi2002" if args.target == "fig6i" else "woodin2003"
    param_space = {
        "dataset": dataset,
        "weighting": args.weighting,
        "variance_floor": float(args.variance_floor),
        "transmission_delay_ms": tune.uniform(0.0, 10.0),
        "decay_rate": tune.loguniform(0.005, 0.5),
        "plasticity_threshold": tune.uniform(-0.5, 0.5),
        "learning_rate": tune.loguniform(0.01, 1.0),
        "initial_weight": tune.uniform(0.0, 2.0),
        "seed": int(args.seed),
    }
    return tune_config, param_space


def _row_value(row: Mapping[str, Any], name: str) -> Any:
    if name in row and row[name] is not None:
        return row[name]
    config_name = f"config/{name}"
    if config_name in row and row[config_name] is not None:
        return row[config_name]
    raise KeyError(f"Ray results do not contain {name!r} or {config_name!r}")


def fit_trials_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip Ray bookkeeping while retaining all fit parameters and metrics."""

    rows: list[dict[str, Any]] = []
    for trial_number, (_, series) in enumerate(frame.iterrows()):
        raw = series.to_dict()
        row: dict[str, Any] = {"trial": trial_number}
        for name in ("objective", "unweighted_sse", "unweighted_rmse"):
            if name in raw and raw[name] is not None:
                row[name] = raw[name]
        for name in fit_stdp.PARAMETER_NAMES:
            row[name] = _row_value(raw, name)
        rows.append(row)
    trials = pd.DataFrame(rows)
    if "objective" not in trials:
        raise RuntimeError("mini-radas results do not contain the objective metric")
    trials["objective"] = pd.to_numeric(trials["objective"], errors="coerce")
    return trials.sort_values("trial", kind="stable").reset_index(drop=True)


def materialize_fit(
    args: argparse.Namespace,
    profile: SearchProfile,
    frame: pd.DataFrame,
    output: Path,
) -> dict[str, Path]:
    trials = fit_trials_frame(frame)
    finite = trials[np.isfinite(trials["objective"].to_numpy(dtype=float))]
    if finite.empty:
        raise RuntimeError("all mini-radas trials returned non-finite objectives")
    best = finite.loc[finite["objective"].idxmin()]
    params = {name: float(best[name]) for name in fit_stdp.PARAMETER_NAMES}
    dataset = "bi2002" if args.target == "fig6i" else "woodin2003"
    data = fit_stdp.load_dataset(dataset)
    output.mkdir(parents=True, exist_ok=True)
    paths = fit_stdp.materialize_result(
        data=data,
        params=params,
        weighting=args.weighting,
        variance_floor=float(args.variance_floor),
        output=output,
        formats=[] if args.no_plots else ["pdf", "svg"],
        dpi=int(args.dpi),
        source="mini-radas_optuna",
        seed=int(args.seed),
        num_samples=profile.num_samples,
    )
    trials_csv = output / f"{data['stem']}_trials.csv"
    trials.to_csv(trials_csv, index=False)
    paths["trials_csv"] = trials_csv
    return paths


def write_metadata(
    *,
    args: argparse.Namespace,
    profile: SearchProfile,
    frame: pd.DataFrame,
    output: Path,
    storage: Path,
    experiment_name: str,
    run_id: str,
    source_sha256: str,
    paths: Mapping[str, Path],
) -> Path:
    payload = {
        "target": args.target,
        "profile": profile.name,
        "run_id": run_id,
        "seed": int(args.seed),
        "experiment_name": experiment_name,
        "execution": "local",
        "orchestrator": {
            "package": "mini-radas",
            "expected_commit": MINI_RADAS_COMMIT,
            "verified_source_sha256": source_sha256,
            "api": "radas.run_experiment",
        },
        "storage": str(storage),
        "output": str(output),
        "completed_trials": int(len(frame)),
        "requested_num_samples": profile.num_samples,
        "files": {key: str(value) for key, value in sorted(paths.items())},
        "options": {
            "weighting": args.weighting,
            "variance_floor": (
                float(args.variance_floor) if args.weighting == "manuscript" else None
            ),
            "cpus_per_trial": float(args.cpus_per_trial),
            "gpu_per_trial": 0,
        },
    }
    metadata = output / f"{args.target}_run.json"
    metadata.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return metadata


async def run(args: argparse.Namespace) -> dict[str, Path]:
    """Run one local public mini-radas search and write portable outputs."""

    profile = resolve_profile(args)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.variance_floor <= 0:
        raise ValueError("--variance-floor must be positive")
    if args.cpus_per_trial <= 0:
        raise ValueError("--cpus-per-trial must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    run_id = run_identifier(args, profile)
    experiment_name = args.experiment_name or f"pdm-{args.target}-{run_id}"
    validate_storage_component(experiment_name, "--experiment-name")
    validate_storage_component(args.user_name, "--user-name")

    # Validate names before importing the orchestrator or creating directories.
    try:
        import radas as radas_module
        from radas import run_experiment
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise RuntimeError(
            "mini-radas is not installed; install requirements-search.txt first."
        ) from exc
    source_sha256 = validate_mini_radas(radas_module, run_experiment)
    tune_config, param_space = build_search(args, profile)
    storage = resolve_path(args.storage)
    output = (
        resolve_path(args.output)
        if args.output is not None
        else (DEFAULT_OUTPUT_ROOT / args.target / run_id).resolve()
    )
    storage.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    original_directory = Path.cwd()
    try:
        os.chdir(HERE)
        result = await run_experiment(
            user_name=args.user_name,
            trainable=fit_stdp.stdp_fit_trainable,
            experiment_name=experiment_name,
            resources={"cpu": float(args.cpus_per_trial), "gpu": 0},
            run_with="local",
            local_storage_path=str(storage),
            tuner_init_kwargs={
                "tune_config": tune_config,
                "run_config_kwargs": {"verbose": 0 if profile.name == "smoke" else 1},
            },
            param_space=param_space,
            dos=["run", "analyze"],
        )
    finally:
        os.chdir(original_directory)

    frame = result.get("df")
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError("mini-radas did not return its analysis dataframe")
    if len(frame) != profile.num_samples:
        raise RuntimeError(
            f"mini-radas returned {len(frame)} trials; expected {profile.num_samples}"
        )
    paths = materialize_fit(args, profile, frame, output)
    paths["run_metadata"] = write_metadata(
        args=args,
        profile=profile,
        frame=frame,
        output=output,
        storage=storage,
        experiment_name=experiment_name,
        run_id=run_id,
        source_sha256=source_sha256,
        paths=paths,
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=TARGETS)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--quick", action="store_true", help="run 32 trials")
    profile.add_argument("--smoke", action="store_true", help="run one trial")
    parser.add_argument(
        "--num-samples", type=int, help="override the profile's number of trials"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--weighting",
        choices=("legacy", "manuscript", "none"),
        default="legacy",
    )
    parser.add_argument("--variance-floor", type=float, default=1e-12)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--user-name", default=DEFAULT_USER_NAME)
    parser.add_argument("--experiment-name")
    parser.add_argument("--cpus-per-trial", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = asyncio.run(run(args))
    print(f"Completed {args.target} with mini-radas.")
    for label, path in sorted(paths.items()):
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
