#!/usr/bin/env python3
"""Run the Supplementary Figure 3d grid with public mini-radas.

The entry point uses only the public ``radas.run_experiment`` API from
mini-radas commit ``8314d91b773db5148b0a90a8442f549e367e9cfb``. Relative
storage and output paths are resolved from this file's directory.
"""

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
    from . import supplementary_figure3d as experiment
except ImportError:  # Support ``python run_with_radas.py`` from this directory.
    import supplementary_figure3d as experiment


HERE = Path(__file__).resolve().parent
MINI_RADAS_COMMIT = "8314d91b773db5148b0a90a8442f549e367e9cfb"
MINI_RADAS_SOURCE_SHA256 = (
    "b0b85dcea68d1992c1cfd5e39725d41e93e8826ed55c332ad20870672191dda4"
)
MINI_RADAS_SOURCE_FILES = ("__init__.py", "core.py", "run_experiment.py", "utils.py")

DEFAULT_STORAGE = HERE / "generated" / "radas_storage"
DEFAULT_OUTPUT_ROOT = HERE / "generated" / "radas"
DEFAULT_USER_NAME = "pdm-public"
SAFE_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

INITIAL_BACKWARD_WEIGHT = 0.5
THETA = 0.4
LEARNING_RATE = 0.005
BACKWARD_LEARNING_RATE = 0.005
DT = 0.1
TMAX = 50.0
DELAY = 5.0
OUTPUT_DELAY_STD = 0.0


@dataclass(frozen=True)
class RunProfile:
    """Workload selected by the full, quick, or smoke profile."""

    name: str
    iterations: int
    grid_values: tuple[float, ...]

    @property
    def expected_trials(self) -> int:
        return len(self.grid_values) ** 2


@dataclass(frozen=True)
class ExperimentSpec:
    """Arguments passed to the public mini-radas function."""

    trainable: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    param_space: dict[str, Any]
    tune_config: Any


def resolve_script_path(path: Path | str) -> Path:
    """Resolve relative paths from this script rather than the caller's CWD."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = HERE / candidate
    return candidate.resolve()


def validate_storage_component(value: str, option: str) -> str:
    """Accept only one conservative path component for mini-radas names."""

    if value in {".", ".."} or SAFE_COMPONENT_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{option} must match {SAFE_COMPONENT_PATTERN.pattern!r} "
            "and cannot be '.' or '..'"
        )
    return value


def _mini_radas_source_fingerprint(module: Any) -> str:
    """Hash the source files distributed by the pinned mini-radas revision."""

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
    """Fail unless the imported package matches the pinned public source."""

    required = {"user_name", "resources", "run_with", "local_storage_path", "dos"}
    missing = sorted(required.difference(inspect.signature(run_experiment).parameters))
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
    """Return the exact publication, development, or integration workload."""

    if args.smoke:
        return RunProfile("smoke", 1, (0.0,))
    if args.quick:
        return RunProfile("quick", 4, (-1.0, 0.0, 1.0))
    return RunProfile(
        "full",
        64,
        tuple(float(value) for value in experiment.grid_values(-1.0, 1.0, 0.25)),
    )


def run_identifier(profile: RunProfile, seed: int) -> str:
    """Build a deterministic identifier that isolates distinct runs."""

    payload = {
        "profile": profile.name,
        "iterations": profile.iterations,
        "grid_values": profile.grid_values,
        "seed": int(seed),
    }
    suffix = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return f"{profile.name}-seed-{seed}-{suffix}"


def _load_tune() -> Any:
    try:
        from ray import tune
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise RuntimeError(
            "Ray Tune is required; install requirements-search.txt."
        ) from exc
    return tune


def build_experiment_spec(profile: RunProfile, seed: int) -> ExperimentSpec:
    """Build the 2-D initial-weight grid and fixed model configuration."""

    tune = _load_tune()
    values = list(profile.grid_values)
    param_space = {
        "initial_wf1": tune.grid_search(values),
        "initial_wf2": tune.grid_search(values),
        "initial_backward_weight": INITIAL_BACKWARD_WEIGHT,
        "theta": THETA,
        "learning_rate": LEARNING_RATE,
        "backward_learning_rate": BACKWARD_LEARNING_RATE,
        "num_dataset_iterations": profile.iterations,
        "seed": int(seed),
        "dt": DT,
        "tmax": TMAX,
        "delay": DELAY,
        "output_delay_std": OUTPUT_DELAY_STD,
    }
    return ExperimentSpec(
        trainable=experiment.spiking_trajectory_trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(),
    )


def _sequence(value: Any, field: str) -> list[Any]:
    """Coerce Ray/pandas representations of a list-valued metric."""

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
    raise TypeError(f"Ray result field {field!r} is not list-valued: {type(value)!r}")


def _row_value(row: Mapping[str, Any], name: str) -> Any:
    """Read a returned metric, falling back to its Ray config column."""

    if name in row and row[name] is not None:
        return row[name]
    config_name = f"config/{name}"
    if config_name in row and row[config_name] is not None:
        return row[config_name]
    raise KeyError(f"Ray results do not contain {name!r} or {config_name!r}")


def _grid_record(row: Mapping[str, Any]) -> dict[str, Any]:
    scalar_columns = experiment.CSV_COLUMNS[
        : experiment.CSV_COLUMNS.index("dataset_iteration")
    ]
    record = {name: _row_value(row, name) for name in scalar_columns}
    for name in ("iterations", "wf1s", "wf2s", "wbs"):
        record[name] = _sequence(_row_value(row, name), name)
    for name in (
        "num_dataset_iterations",
        "sample_updates",
        "final_wf1",
        "final_wf2",
        "final_wb",
        "weight_displacement",
        "last_output",
        "last_dendritic_error",
    ):
        record[name] = _row_value(row, name)
    return record


def _write_dict_rows(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def materialize_results(
    frame: pd.DataFrame, output: Path, *, plots: bool
) -> dict[str, Path]:
    """Write long trajectories, one-row trial summaries, and optional plots."""

    if frame.empty:
        raise RuntimeError("mini-radas returned no completed trials")
    output.mkdir(parents=True, exist_ok=True)

    trajectories: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for trial_number, (_, series) in enumerate(frame.iterrows()):
        record = _grid_record(series.to_dict())
        trajectories.extend(experiment.trajectory_rows(record))
        trials.append(
            {
                "trial": trial_number,
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

    trajectories.sort(
        key=lambda row: (
            float(row["initial_wf1"]),
            float(row["initial_wf2"]),
            int(row["dataset_iteration"]),
        )
    )
    trials.sort(key=lambda row: (row["initial_wf1"], row["initial_wf2"]))

    stem = "supplementary_figure3d"
    trajectories_csv = output / f"{stem}.csv"
    trials_csv = output / f"{stem}_trials.csv"
    _write_dict_rows(trajectories_csv, experiment.CSV_COLUMNS, trajectories)
    _write_dict_rows(trials_csv, tuple(trials[0]), trials)

    paths = {"trajectories_csv": trajectories_csv, "trials_csv": trials_csv}
    if plots:
        pdf_path = output / f"{stem}.pdf"
        svg_path = output / f"{stem}.svg"
        experiment.plot_results(
            trajectories_csv,
            pdf_path=pdf_path,
            svg_path=svg_path,
        )
        paths.update({"pdf": pdf_path, "svg": svg_path})
    return paths


def validate_trial_count(frame: pd.DataFrame, profile: RunProfile) -> None:
    """Require one successful result row for every requested grid point."""

    if len(frame) != profile.expected_trials:
        raise RuntimeError(
            f"mini-radas returned {len(frame)} trials; "
            f"expected {profile.expected_trials}"
        )


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_metadata(
    *,
    args: argparse.Namespace,
    profile: RunProfile,
    frame: pd.DataFrame,
    output: Path,
    storage: Path,
    experiment_name: str,
    run_id: str,
    source_sha256: str,
    paths: Mapping[str, Path],
) -> Path:
    payload = {
        "panel": "Supplementary Figure 3d",
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
        "expected_trials": profile.expected_trials,
        "dataset_iterations": profile.iterations,
        "grid": {
            "values": list(profile.grid_values),
            "number_of_initial_conditions": profile.expected_trials,
        },
        "fixed_parameters": {
            "initial_backward_weight": INITIAL_BACKWARD_WEIGHT,
            "theta": THETA,
            "learning_rate": LEARNING_RATE,
            "backward_learning_rate": BACKWARD_LEARNING_RATE,
            "dt": DT,
            "tmax": TMAX,
            "delay": DELAY,
            "output_delay_std": OUTPUT_DELAY_STD,
        },
        "resources": {"cpus_per_trial": float(args.cpus_per_trial), "gpu": 0},
        "files": {key: str(value) for key, value in sorted(paths.items())},
    }
    metadata = output / "supplementary_figure3d_run.json"
    metadata.write_text(
        json.dumps(payload, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )
    return metadata


async def run(args: argparse.Namespace) -> dict[str, Path]:
    """Execute a local mini-radas grid and materialize publication outputs."""

    profile = resolve_profile(args)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.cpus_per_trial <= 0:
        raise ValueError("--cpus-per-trial must be positive")
    run_id = run_identifier(profile, args.seed)
    experiment_name = args.experiment_name or f"pdm-supp3d-{run_id}"
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
    spec = build_experiment_spec(profile, args.seed)

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
    previous_cwd = Path.cwd()
    try:
        os.chdir(HERE)
        result = await run_experiment(
            user_name=args.user_name,
            trainable=spec.trainable,
            experiment_name=experiment_name,
            resources={"cpu": float(args.cpus_per_trial), "gpu": 0},
            run_with="local",
            local_storage_path=str(storage),
            tuner_init_kwargs={
                "tune_config": spec.tune_config,
                "run_config_kwargs": {"verbose": 0 if profile.name == "smoke" else 1},
            },
            param_space=spec.param_space,
            dos=["run", "analyze"],
        )
    finally:
        os.chdir(previous_cwd)

    frame = result.get("df")
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError("mini-radas did not return its analysis dataframe")
    validate_trial_count(frame, profile)

    paths = materialize_results(frame, output, plots=not args.no_plots)
    paths["run_metadata"] = _write_metadata(
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
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument(
        "--quick",
        action="store_true",
        help="run 4 iterations on a 3x3 grid",
    )
    profile.add_argument(
        "--smoke",
        action="store_true",
        help="run 1 iteration at the single point (0, 0)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--storage",
        type=Path,
        default=DEFAULT_STORAGE,
        help="Ray storage root; relative paths are script-relative",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory; relative paths are script-relative",
    )
    parser.add_argument("--user-name", default=DEFAULT_USER_NAME)
    parser.add_argument("--experiment-name")
    parser.add_argument("--cpus-per-trial", type=float, default=1.0)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = asyncio.run(run(args))
    print("Completed Supplementary Figure 3d with mini-radas.")
    for label, path in sorted(paths.items()):
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
