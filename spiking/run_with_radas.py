#!/usr/bin/env python3
"""Run publication experiments locally with the open-source mini-radas API.

This entry point targets ``mini-radas`` at commit
``8314d91b773db5148b0a90a8442f549e367e9cfb``.  It deliberately uses only
the public ``radas.run_experiment`` interface and local Ray execution.

Examples
--------
Run a complete paper grid or a 1,000-trial fit::

    python spiking/run_with_radas.py supp3c
    python spiking/run_with_radas.py fig6i --num-samples 1000

Perform a small integration check::

    python spiking/run_with_radas.py supp3d --smoke --no-plots

Relative ``--storage`` and ``--output`` paths are resolved from this file's
directory, so commands behave the same way regardless of the caller's current
working directory.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import hashlib
import inspect
import json
import math
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
REPOSITORY_ROOT = HERE.parent
XOR_DIR = REPOSITORY_ROOT / "xor"

# Package imports make the trainables importable in local Ray worker processes.
# The explicit repository path also supports ``python /path/to/run_with_radas.py``.
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from spiking import fit_initial_strength, fit_stdp  # noqa: E402
from spiking.supplementary_figure3d import (  # noqa: E402
    CSV_COLUMNS as SPIKING_GRID_COLUMNS,
    grid_values as spiking_grid_values,
    plot_results as plot_spiking_trajectories,
    spiking_trajectory_trainable,
    trajectory_rows as spiking_trajectory_rows,
)
from xor.plot_weight_trajectories import plot_trajectories as plot_rate_trajectories  # noqa: E402
from xor.weight_trajectories import (  # noqa: E402
    CSV_COLUMNS as RATE_GRID_COLUMNS,
    firing_rate_trajectory_trainable,
    grid_values as rate_grid_values,
    trajectory_rows as rate_trajectory_rows,
)

MINI_RADAS_COMMIT = "8314d91b773db5148b0a90a8442f549e367e9cfb"
MINI_RADAS_SOURCE_SHA256 = (
    "b0b85dcea68d1992c1cfd5e39725d41e93e8826ed55c332ad20870672191dda4"
)
MINI_RADAS_SOURCE_FILES = ("__init__.py", "core.py", "run_experiment.py", "utils.py")
TARGETS = ("supp3c", "supp3d", "fig6i", "fig6j", "fig7b")
FIT_TARGETS = frozenset(("fig6i", "fig6j", "fig7b"))
GRID_TARGETS = frozenset(("supp3c", "supp3d"))

DEFAULT_STORAGE = HERE / "generated" / "radas_storage"
DEFAULT_OUTPUT_ROOT = HERE / "generated" / "radas"
DEFAULT_USER_NAME = "pdm-public"
SAFE_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True)
class RunProfile:
    """Resolved amount of work for one CLI invocation."""

    name: str
    num_samples: int | None
    iterations: int | None
    grid_min: float | None
    grid_max: float | None
    grid_step: float | None


@dataclass(frozen=True)
class ExperimentSpec:
    """Arguments needed by ``radas.run_experiment`` and materialization."""

    trainable: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    param_space: dict[str, Any]
    tune_config: Any
    metric: str | None
    mode: str | None


def resolve_script_path(path: Path | str) -> Path:
    """Resolve a path relative to this script, without requiring it to exist."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = HERE / candidate
    return candidate.resolve()


def validate_storage_component(value: str, option: str) -> str:
    """Validate a user-controlled name used below the Ray storage root.

    mini-radas treats both values as directory names.  Limiting them to one
    conservative ASCII component prevents separators, traversal components,
    shell metacharacters, and platform-specific path syntax from reaching its
    replacement/removal logic.
    """

    if value in {".", ".."} or SAFE_COMPONENT_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{option} must match {SAFE_COMPONENT_PATTERN.pattern!r} "
            "and cannot be '.' or '..'"
        )
    return value


def _mini_radas_source_fingerprint(module: Any) -> str:
    """Hash the public package files shipped by the pinned source revision."""

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
    """Fail closed unless the imported code is the pinned public mini-radas."""

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


def profile_name(args: argparse.Namespace) -> str:
    if args.smoke:
        return "smoke"
    if args.quick:
        return "quick"
    return "full"


def resolve_profile(args: argparse.Namespace) -> RunProfile:
    """Resolve full/quick/smoke defaults, retaining explicit overrides."""

    name = profile_name(args)
    if args.target in FIT_TARGETS:
        defaults = {"full": 1000, "quick": 32, "smoke": 1}
        num_samples = (
            args.num_samples if args.num_samples is not None else defaults[name]
        )
        if num_samples < 1:
            raise ValueError("--num-samples must be at least 1")
        return RunProfile(name, int(num_samples), None, None, None, None)

    iteration_defaults = {
        "supp3c": {"full": 128, "quick": 8, "smoke": 1},
        "supp3d": {"full": 64, "quick": 4, "smoke": 1},
    }
    iterations = (
        args.iterations
        if args.iterations is not None
        else iteration_defaults[args.target][name]
    )
    if iterations < 1:
        raise ValueError("--iterations must be at least 1")

    default_step = 0.25 if name == "full" else 1.0
    grid_min = -1.0 if args.grid_min is None else args.grid_min
    grid_max = 1.0 if args.grid_max is None else args.grid_max
    grid_step = default_step if args.grid_step is None else args.grid_step
    if grid_step <= 0:
        raise ValueError("--grid-step must be positive")
    if grid_max < grid_min:
        raise ValueError("--grid-max must be greater than or equal to --grid-min")
    return RunProfile(
        name,
        None,
        int(iterations),
        float(grid_min),
        float(grid_max),
        float(grid_step),
    )


def run_identifier(args: argparse.Namespace, profile: RunProfile) -> str:
    """Return a readable, collision-resistant identifier for result isolation."""

    payload = {
        "target": args.target,
        "profile": {
            "name": profile.name,
            "num_samples": profile.num_samples,
            "iterations": profile.iterations,
            "grid_min": profile.grid_min,
            "grid_max": profile.grid_max,
            "grid_step": profile.grid_step,
        },
        "seed": int(args.seed),
        "weighting": args.weighting if args.target in {"fig6i", "fig6j"} else None,
        "variance_floor": (
            float(args.variance_floor) if args.target in {"fig6i", "fig6j"} else None
        ),
        "search_space": args.search_space if args.target == "fig7b" else None,
    }
    suffix = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    return f"{profile.name}-seed-{args.seed}-{suffix}"


def _load_tune_dependencies() -> tuple[Any, Any]:
    """Import Ray Tune and its public Optuna adapter with a useful error."""

    try:
        from ray import tune
        from ray.tune.search.optuna import OptunaSearch
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "Ray Tune and Optuna are required; install spiking/requirements.txt first."
        ) from exc
    return tune, OptunaSearch


def _grid_values(target: str, profile: RunProfile) -> list[float]:
    if profile.name == "smoke":
        return [0.0]
    assert profile.grid_min is not None
    assert profile.grid_max is not None
    assert profile.grid_step is not None
    factory = rate_grid_values if target == "supp3c" else spiking_grid_values
    return [
        float(value)
        for value in factory(profile.grid_min, profile.grid_max, profile.grid_step)
    ]


def build_experiment_spec(
    args: argparse.Namespace, profile: RunProfile
) -> ExperimentSpec:
    """Build the Ray search objects consumed by the public orchestrator API."""

    tune, optuna_search_type = _load_tune_dependencies()
    if args.target in GRID_TARGETS:
        values = _grid_values(args.target, profile)
        param_space: dict[str, Any] = {
            "initial_wf1": tune.grid_search(values),
            "initial_wf2": tune.grid_search(values),
            "num_dataset_iterations": int(profile.iterations),
            "seed": int(args.seed),
        }
        if args.target == "supp3c":
            param_space.update(
                {
                    "initial_backward_weight": 0.5,
                    "learning_rate": 0.1,
                    "backward_learning_rate": 0.1,
                    "theta_forward": 0.5,
                    "theta_backward": 0.5,
                }
            )
            trainable = firing_rate_trajectory_trainable
        else:
            param_space.update(
                {
                    "initial_backward_weight": 0.5,
                    "theta": 0.4,
                    "learning_rate": 0.005,
                    "backward_learning_rate": 0.005,
                    "dt": 0.1,
                    "tmax": 50.0,
                    "delay": 5.0,
                    "output_delay_std": 0.0,
                }
            )
            trainable = spiking_trajectory_trainable
        # Grid trials are independent and each carries its own seed, so local
        # scheduling order cannot change the numerical result.
        tune_config = tune.TuneConfig()
        return ExperimentSpec(trainable, param_space, tune_config, None, None)

    metric = "objective"
    mode = "min"
    search_algorithm = optuna_search_type(metric=metric, mode=mode, seed=args.seed)
    tune_config = tune.TuneConfig(
        metric=metric,
        mode=mode,
        search_alg=search_algorithm,
        num_samples=int(profile.num_samples),
        max_concurrent_trials=1,
    )
    if args.target in {"fig6i", "fig6j"}:
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
        trainable = fit_stdp.stdp_fit_trainable
    else:
        bounds = fit_initial_strength._search_bounds(args.search_space)
        param_space = {
            name: (
                tune.loguniform(low, high) if logarithmic else tune.uniform(low, high)
            )
            for name, (low, high, logarithmic) in bounds.items()
        }
        param_space.update(
            {
                "search_space": args.search_space,
                "seed": int(args.seed),
            }
        )
        trainable = fit_initial_strength.initial_strength_fit_trainable
    return ExperimentSpec(trainable, param_space, tune_config, metric, mode)


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
    """Prefer a returned metric, then fall back to its Ray config column."""

    if name in row and row[name] is not None:
        return row[name]
    config_name = f"config/{name}"
    if config_name in row and row[config_name] is not None:
        return row[config_name]
    raise KeyError(f"Ray results do not contain {name!r} or {config_name!r}")


def _write_dict_rows(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _grid_record(
    row: Mapping[str, Any], scalar_columns: Sequence[str]
) -> dict[str, Any]:
    record = {name: _row_value(row, name) for name in scalar_columns}
    for name in ("iterations", "wf1s", "wf2s", "wbs"):
        record[name] = _sequence(_row_value(row, name), name)
    for name in (
        "num_dataset_iterations",
        "optimizer_steps",
        "sample_updates",
        "final_wf1",
        "final_wf2",
        "final_wb",
        "weight_displacement",
        "last_output",
        "last_dendritic_error",
    ):
        try:
            record[name] = _row_value(row, name)
        except KeyError:
            pass
    return record


def _materialize_grid(
    target: str,
    frame: pd.DataFrame,
    output: Path,
    *,
    plots: bool,
) -> dict[str, Path]:
    """Explode one row per Ray trial into paper-ready trajectory tables."""

    if frame.empty:
        raise RuntimeError("mini-radas returned no completed grid trials")
    output.mkdir(parents=True, exist_ok=True)
    if target == "supp3c":
        columns = RATE_GRID_COLUMNS
        rows_from_record = rate_trajectory_rows
        stem = "supplementary_figure3c"
    else:
        columns = SPIKING_GRID_COLUMNS
        rows_from_record = spiking_trajectory_rows
        stem = "supplementary_figure3d"

    scalar_columns = columns[: columns.index("dataset_iteration")]
    long_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for row_number, (_, series) in enumerate(frame.iterrows()):
        record = _grid_record(series.to_dict(), scalar_columns)
        long_rows.extend(rows_from_record(record))
        summaries.append(
            {
                "trial": row_number,
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
    trajectories_csv = output / f"{stem}.csv"
    trials_csv = output / f"{stem}_trials.csv"
    _write_dict_rows(trajectories_csv, columns, long_rows)
    _write_dict_rows(trials_csv, tuple(summaries[0]), summaries)

    paths = {"trajectories_csv": trajectories_csv, "trials_csv": trials_csv}
    if plots:
        pdf_path = output / f"{stem}.pdf"
        svg_path = output / f"{stem}.svg"
        if target == "supp3c":
            plot_rate_trajectories(
                trajectories_csv,
                pdf_path=pdf_path,
                svg_path=svg_path,
            )
        else:
            plot_spiking_trajectories(
                trajectories_csv,
                pdf_path=pdf_path,
                svg_path=svg_path,
            )
        paths.update({"pdf": pdf_path, "svg": svg_path})
    return paths


def _parameter_columns(target: str) -> tuple[str, ...]:
    if target in {"fig6i", "fig6j"}:
        return tuple(fit_stdp.PARAMETER_NAMES)
    return tuple(fit_initial_strength.PARAMETER_NAMES)


def _fit_trials_frame(target: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable, concise fit table independent of Ray bookkeeping."""

    parameters = _parameter_columns(target)
    rows: list[dict[str, Any]] = []
    for trial_number, (_, series) in enumerate(frame.iterrows()):
        raw = series.to_dict()
        row: dict[str, Any] = {"trial": trial_number}
        for name in ("objective", "unweighted_sse", "unweighted_rmse", "rmse"):
            if name in raw and raw[name] is not None:
                row[name] = raw[name]
        for name in parameters:
            row[name] = _row_value(raw, name)
        rows.append(row)
    trials = pd.DataFrame(rows)
    if "objective" not in trials:
        raise RuntimeError("mini-radas results do not contain the objective metric")
    trials["objective"] = pd.to_numeric(trials["objective"], errors="coerce")
    return trials.sort_values("trial", kind="stable").reset_index(drop=True)


def _materialize_fit(
    args: argparse.Namespace,
    profile: RunProfile,
    frame: pd.DataFrame,
    output: Path,
    *,
    plots: bool,
) -> dict[str, Path]:
    """Select the best Ray trial and render its final data products."""

    trials = _fit_trials_frame(args.target, frame)
    finite = trials[np.isfinite(trials["objective"].to_numpy(dtype=float))]
    if finite.empty:
        raise RuntimeError("all mini-radas fit trials returned non-finite objectives")
    best = finite.loc[finite["objective"].idxmin()]
    params = {name: float(best[name]) for name in _parameter_columns(args.target)}
    output.mkdir(parents=True, exist_ok=True)

    if args.target in {"fig6i", "fig6j"}:
        dataset = "bi2002" if args.target == "fig6i" else "woodin2003"
        data = fit_stdp.load_dataset(dataset)
        paths = fit_stdp.materialize_result(
            data=data,
            params=params,
            weighting=args.weighting,
            variance_floor=float(args.variance_floor),
            output=output,
            formats=[] if not plots else ["pdf", "svg"],
            dpi=int(args.dpi),
            source="mini-radas_optuna",
            seed=int(args.seed),
            num_samples=int(profile.num_samples),
        )
        stem = str(data["stem"])
    else:
        paths = fit_initial_strength.materialize_result(
            data=fit_initial_strength.load_dataset(),
            params=params,
            output=output,
            formats=[] if not plots else ["pdf", "svg"],
            dpi=int(args.dpi),
            source="mini-radas_optuna",
            search_space=args.search_space,
            seed=int(args.seed),
            num_samples=int(profile.num_samples),
        )
        stem = "fig7b_initial_strength"

    trials_csv = output / f"{stem}_trials.csv"
    trials.to_csv(trials_csv, index=False)
    paths["trials_csv"] = trials_csv
    return paths


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_run_metadata(
    args: argparse.Namespace,
    profile: RunProfile,
    frame: pd.DataFrame,
    output: Path,
    experiment_name: str,
    storage: Path,
    paths: Mapping[str, Path],
    source_sha256: str,
    run_id: str,
) -> Path:
    grid_values = (
        _grid_values(args.target, profile) if args.target in GRID_TARGETS else None
    )
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
        "dataset_iterations": profile.iterations,
        "grid": (
            {
                "values": grid_values,
                "number_of_initial_conditions": len(grid_values) ** 2,
            }
            if grid_values is not None
            else None
        ),
        "files": {key: str(value) for key, value in sorted(paths.items())},
        "options": {
            "weighting": args.weighting if args.target in {"fig6i", "fig6j"} else None,
            "variance_floor": (
                float(args.variance_floor)
                if args.target in {"fig6i", "fig6j"}
                else None
            ),
            "search_space": args.search_space if args.target == "fig7b" else None,
            "cpus_per_trial": float(args.cpus_per_trial),
            "gpu_per_trial": 0,
            "model_constants": (
                {
                    "decay_per_ms": 0.1,
                    "spike_threshold": 0.5,
                    "reset_potential": -0.5,
                    "hidden_gate_epsilon": 0.05,
                    "output_gate_epsilon": -1.0,
                    "gate_gain": 200.0,
                }
                if args.target == "supp3d"
                else None
            ),
        },
    }
    metadata_path = output / f"{args.target}_run.json"
    metadata_path.write_text(
        json.dumps(payload, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )
    return metadata_path


async def run(args: argparse.Namespace) -> dict[str, Path]:
    """Execute one local mini-radas experiment and materialize its outputs."""

    profile = resolve_profile(args)
    if args.target in GRID_TARGETS and args.num_samples is not None:
        raise ValueError("--num-samples applies only to fig6i, fig6j, and fig7b")
    if args.target in FIT_TARGETS and any(
        value is not None
        for value in (args.iterations, args.grid_min, args.grid_max, args.grid_step)
    ):
        raise ValueError("--iterations and --grid-* apply only to supp3c and supp3d")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    run_id = run_identifier(args, profile)
    experiment_name = args.experiment_name or f"pdm-{args.target}-{run_id}"
    validate_storage_component(experiment_name, "--experiment-name")
    validate_storage_component(args.user_name, "--user-name")
    if args.cpus_per_trial <= 0:
        raise ValueError("--cpus-per-trial must be positive")

    # Perform all path-component validation before importing or invoking the
    # orchestrator and before creating storage/output directories.
    try:
        import radas as radas_module
        from radas import run_experiment
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "mini-radas is not installed; install spiking/requirements.txt first."
        ) from exc
    source_sha256 = validate_mini_radas(radas_module, run_experiment)
    spec = build_experiment_spec(args, profile)
    storage = resolve_script_path(args.storage)
    output = (
        resolve_script_path(args.output)
        if args.output is not None
        else (DEFAULT_OUTPUT_ROOT / args.target / run_id).resolve()
    )
    storage.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    original_working_directory = Path.cwd()
    try:
        # Local Ray workers inherit this importable repository working directory.
        os.chdir(REPOSITORY_ROOT)
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
        os.chdir(original_working_directory)

    frame = result.get("df")
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError("mini-radas did not return its analysis dataframe")
    expected_trials = (
        len(_grid_values(args.target, profile)) ** 2
        if args.target in GRID_TARGETS
        else int(profile.num_samples)
    )
    if len(frame) != expected_trials:
        raise RuntimeError(
            f"mini-radas returned {len(frame)} trials; expected {expected_trials}"
        )
    if args.target in GRID_TARGETS:
        paths = _materialize_grid(
            args.target,
            frame,
            output,
            plots=not args.no_plots,
        )
    else:
        paths = _materialize_fit(
            args,
            profile,
            frame,
            output,
            plots=not args.no_plots,
        )
    metadata = _write_run_metadata(
        args,
        profile,
        frame,
        output,
        experiment_name,
        storage,
        paths,
        source_sha256,
        run_id,
    )
    paths["run_metadata"] = metadata
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=TARGETS, help="paper panel/search to run")
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument(
        "--quick",
        action="store_true",
        help="reduced 3x3 grid or 32-trial fit for development",
    )
    profile.add_argument(
        "--smoke",
        action="store_true",
        help="one initial condition/iteration or one fit trial",
    )
    parser.add_argument("--seed", type=int, default=0, help="deterministic seed")
    parser.add_argument(
        "--num-samples",
        type=int,
        help="fit trials (default: 1000 full, 32 quick, 1 smoke)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="dataset iterations for supplementary grids (overrides profile)",
    )
    parser.add_argument("--grid-min", type=float, help="grid lower bound")
    parser.add_argument("--grid-max", type=float, help="grid upper bound")
    parser.add_argument("--grid-step", type=float, help="grid step")
    parser.add_argument(
        "--weighting",
        choices=("legacy", "manuscript", "none"),
        default="legacy",
        help="Fig. 6i-j objective weighting",
    )
    parser.add_argument(
        "--variance-floor",
        type=float,
        default=1e-12,
        help="positive variance floor for manuscript weighting",
    )
    parser.add_argument(
        "--search-space",
        choices=("legacy", "manuscript"),
        default="legacy",
        help="Fig. 7b mapping-range search space",
    )
    parser.add_argument(
        "--storage",
        type=Path,
        default=DEFAULT_STORAGE,
        help="Ray storage root; relative paths are script-relative",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="materialized output directory; relative paths are script-relative",
    )
    parser.add_argument("--user-name", default=DEFAULT_USER_NAME)
    parser.add_argument("--experiment-name")
    parser.add_argument("--cpus-per-trial", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.variance_floor <= 0:
        raise ValueError("--variance-floor must be positive")
    paths = asyncio.run(run(args))
    print(f"Completed {args.target} with mini-radas.")
    for label, path in sorted(paths.items()):
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
