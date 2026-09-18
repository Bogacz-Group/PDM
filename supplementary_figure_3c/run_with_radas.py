#!/usr/bin/env python3
"""Run the Supplementary Figure 3c grid with public mini-radas."""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import hashlib
import inspect
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from .plot_weight_trajectories import plot_trajectories
    from .weight_trajectories import (
        CSV_COLUMNS,
        firing_rate_trajectory_trainable,
        grid_values,
        trajectory_rows,
    )
except ImportError:  # Support ``python run_with_radas.py`` from any directory.
    from plot_weight_trajectories import plot_trajectories
    from weight_trajectories import (
        CSV_COLUMNS,
        firing_rate_trajectory_trainable,
        grid_values,
        trajectory_rows,
    )

HERE = Path(__file__).resolve().parent
DEFAULT_STORAGE = HERE / "generated" / "radas_storage"
DEFAULT_OUTPUT_ROOT = HERE / "generated" / "radas"
DEFAULT_USER_NAME = "pdm-public"
DEFAULT_ITERATIONS = 128

MINI_RADAS_COMMIT = "8314d91b773db5148b0a90a8442f549e367e9cfb"
MINI_RADAS_SOURCE_SHA256 = (
    "b0b85dcea68d1992c1cfd5e39725d41e93e8826ed55c332ad20870672191dda4"
)
MINI_RADAS_SOURCE_FILES = ("__init__.py", "core.py", "run_experiment.py", "utils.py")
SAFE_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True)
class RunProfile:
    """Resolved grid size for one invocation."""

    name: str
    iterations: int
    grid_min: float
    grid_max: float
    grid_step: float


def resolve_script_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = HERE / candidate
    return candidate.resolve()


def validate_storage_component(value: str, option: str) -> str:
    """Reject traversal and unsafe names passed to mini-radas storage code."""

    if value in {".", ".."} or SAFE_COMPONENT_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{option} must match {SAFE_COMPONENT_PATTERN.pattern!r} "
            "and cannot be '.' or '..'"
        )
    return value


def _mini_radas_source_fingerprint(module: Any) -> str:
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
    """Fail closed unless the imported package is the pinned public source."""

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


def resolve_profile(args: argparse.Namespace) -> RunProfile:
    name = "smoke" if args.smoke else "quick" if args.quick else "full"
    defaults = {"full": DEFAULT_ITERATIONS, "quick": 8, "smoke": 1}
    iterations = defaults[name] if args.iterations is None else args.iterations
    if iterations < 1:
        raise ValueError("--iterations must be at least 1")
    minimum = -1.0 if args.grid_min is None else args.grid_min
    maximum = 1.0 if args.grid_max is None else args.grid_max
    step = (0.25 if name == "full" else 1.0) if args.grid_step is None else args.grid_step
    if step <= 0:
        raise ValueError("--grid-step must be positive")
    if maximum < minimum:
        raise ValueError("--grid-max must be greater than or equal to --grid-min")
    return RunProfile(name, int(iterations), float(minimum), float(maximum), float(step))


def profile_grid(profile: RunProfile) -> list[float]:
    if profile.name == "smoke":
        return [0.0]
    return [
        float(value)
        for value in grid_values(profile.grid_min, profile.grid_max, profile.grid_step)
    ]


def run_identifier(args: argparse.Namespace, profile: RunProfile) -> str:
    payload = {
        "profile": profile.name,
        "iterations": profile.iterations,
        "grid": [profile.grid_min, profile.grid_max, profile.grid_step],
        "seed": int(args.seed),
    }
    suffix = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return f"{profile.name}-seed-{args.seed}-{suffix}"


def _sequence(value: Any, field: str) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    raise TypeError(f"Ray result field {field!r} is not list-valued")


def _row_value(row: Mapping[str, Any], name: str) -> Any:
    if name in row and row[name] is not None:
        return row[name]
    config_name = f"config/{name}"
    if config_name in row and row[config_name] is not None:
        return row[config_name]
    raise KeyError(f"Ray results do not contain {name!r} or {config_name!r}")


def _record(row: Mapping[str, Any]) -> dict[str, Any]:
    scalar_columns = CSV_COLUMNS[: CSV_COLUMNS.index("dataset_iteration")]
    record = {name: _row_value(row, name) for name in scalar_columns}
    for name in ("iterations", "wf1s", "wf2s", "wbs"):
        record[name] = _sequence(_row_value(row, name), name)
    for name in (
        "optimizer_steps",
        "sample_updates",
        "final_wf1",
        "final_wf2",
        "final_wb",
        "weight_displacement",
    ):
        record[name] = _row_value(row, name)
    return record


def _write_rows(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def materialize(frame: pd.DataFrame, output: Path, *, plots: bool) -> dict[str, Path]:
    """Write stable, paper-facing files from one row per Ray trial."""

    if frame.empty:
        raise RuntimeError("mini-radas returned no completed grid trials")
    long_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for trial, (_, series) in enumerate(frame.iterrows()):
        record = _record(series.to_dict())
        long_rows.extend(trajectory_rows(record))
        summaries.append(
            {
                "trial": trial,
                "seed": int(record["seed"]),
                "initial_wf1": float(record["initial_wf1"]),
                "initial_wf2": float(record["initial_wf2"]),
                "final_wf1": float(record["final_wf1"]),
                "final_wf2": float(record["final_wf2"]),
                "final_wb": float(record["final_wb"]),
                "weight_displacement": float(record["weight_displacement"]),
                "sample_updates": int(record["sample_updates"]),
            }
        )
    long_rows.sort(
        key=lambda row: (
            float(row["initial_wf1"]),
            float(row["initial_wf2"]),
            int(row["dataset_iteration"]),
        )
    )
    summaries.sort(key=lambda row: (row["initial_wf1"], row["initial_wf2"]))
    trajectories_csv = output / "supplementary_figure3c.csv"
    trials_csv = output / "supplementary_figure3c_trials.csv"
    _write_rows(trajectories_csv, CSV_COLUMNS, long_rows)
    _write_rows(trials_csv, tuple(summaries[0]), summaries)
    paths = {"trajectories_csv": trajectories_csv, "trials_csv": trials_csv}
    if plots:
        pdf_path = output / "supplementary_figure3c.pdf"
        svg_path = output / "supplementary_figure3c.svg"
        plot_trajectories(trajectories_csv, pdf_path=pdf_path, svg_path=svg_path)
        paths.update({"pdf": pdf_path, "svg": svg_path})
    return paths


def write_metadata(
    *,
    args: argparse.Namespace,
    profile: RunProfile,
    output: Path,
    storage: Path,
    experiment_name: str,
    source_sha256: str,
    completed_trials: int,
    paths: Mapping[str, Path],
    run_id: str,
) -> Path:
    values = profile_grid(profile)
    payload = {
        "panel": "Supplementary Figure 3c",
        "profile": profile.name,
        "run_id": run_id,
        "seed": int(args.seed),
        "execution": "local",
        "experiment_name": experiment_name,
        "orchestrator": {
            "package": "mini-radas",
            "expected_commit": MINI_RADAS_COMMIT,
            "verified_source_sha256": source_sha256,
            "api": "radas.run_experiment",
        },
        "storage": str(storage),
        "output": str(output),
        "completed_trials": completed_trials,
        "dataset_iterations": profile.iterations,
        "grid": {"values": values, "number_of_initial_conditions": len(values) ** 2},
        "files": {name: str(path) for name, path in sorted(paths.items())},
        "cpus_per_trial": float(args.cpus_per_trial),
    }
    path = output / "supp3c_run.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


async def run(args: argparse.Namespace) -> dict[str, Path]:
    profile = resolve_profile(args)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.cpus_per_trial <= 0:
        raise ValueError("--cpus-per-trial must be positive")
    run_id = run_identifier(args, profile)
    experiment_name = args.experiment_name or f"pdm-supp3c-{run_id}"
    validate_storage_component(experiment_name, "--experiment-name")
    validate_storage_component(args.user_name, "--user-name")

    try:
        import radas as radas_module
        from radas import run_experiment
        from ray import tune
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise RuntimeError(
            "Install this directory's requirements-search.txt first"
        ) from exc
    source_sha256 = validate_mini_radas(radas_module, run_experiment)
    values = profile_grid(profile)
    param_space = {
        "initial_wf1": tune.grid_search(values),
        "initial_wf2": tune.grid_search(values),
        "initial_backward_weight": 0.5,
        "learning_rate": 0.1,
        "backward_learning_rate": 0.1,
        "theta_forward": 0.5,
        "theta_backward": 0.5,
        "num_dataset_iterations": profile.iterations,
        "seed": int(args.seed),
    }
    storage = resolve_script_path(args.storage)
    output = (
        resolve_script_path(args.output)
        if args.output is not None
        else (DEFAULT_OUTPUT_ROOT / run_id).resolve()
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
            trainable=firing_rate_trajectory_trainable,
            experiment_name=experiment_name,
            resources={"cpu": float(args.cpus_per_trial), "gpu": 0},
            run_with="local",
            local_storage_path=str(storage),
            tuner_init_kwargs={
                "tune_config": tune.TuneConfig(),
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
    expected_trials = len(values) ** 2
    if len(frame) != expected_trials:
        raise RuntimeError(
            f"mini-radas returned {len(frame)} trials; expected {expected_trials}"
        )
    paths = materialize(frame, output, plots=not args.no_plots)
    paths["run_metadata"] = write_metadata(
        args=args,
        profile=profile,
        output=output,
        storage=storage,
        experiment_name=experiment_name,
        source_sha256=source_sha256,
        completed_trials=len(frame),
        paths=paths,
        run_id=run_id,
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--quick", action="store_true", help="3x3 grid, 8 iterations")
    profile.add_argument("--smoke", action="store_true", help="one trial and iteration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--grid-min", type=float)
    parser.add_argument("--grid-max", type=float)
    parser.add_argument("--grid-step", type=float)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--user-name", default=DEFAULT_USER_NAME)
    parser.add_argument("--experiment-name")
    parser.add_argument("--cpus-per-trial", type=float, default=1.0)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = asyncio.run(run(args))
    print("Completed Supplementary Figure 3c with mini-radas.")
    for label, path in sorted(paths.items()):
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
