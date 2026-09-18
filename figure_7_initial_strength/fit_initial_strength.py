"""Evaluate or refit the initial-strength dependence in Figure 7b."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if __package__:
    from .model import stdp_single
else:
    sys.path.insert(0, str(HERE))
    from model import stdp_single

EMBEDDED_BEST_CONFIG: dict[str, float] = {
    "model_initial_weight_start": 0.0841449784793923,
    "model_initial_weight_range": 0.6515770896337575,
    "transmission_delay_ms": 6.7220950525403484,
    "decay_rate": 0.4385945612789581,
    "plasticity_threshold": -0.27095425772257387,
    "learning_rate": 0.13087424482121832,
}
PARAMETER_NAMES = tuple(EMBEDDED_BEST_CONFIG)
CONDITIONS = (
    {
        "name": "pre_post_5ms",
        "label": "Pre-post (+5 ms)",
        "filename": "positive_spiking.csv",
        "delay_ms": 5.0,
    },
    {
        "name": "post_pre_6ms",
        "label": "Post-pre (-6 ms)",
        "filename": "negative_spiking.csv",
        "delay_ms": -6.0,
    },
)


def load_dataset() -> dict[str, Any]:
    """Load the digitized Bi and Poo (1998) conditions bundled here."""

    names: list[str] = []
    labels: list[str] = []
    delays: list[float] = []
    amplitudes: list[float] = []
    changes: list[float] = []
    for condition in CONDITIONS:
        values = np.atleast_2d(
            np.loadtxt(
                HERE / "data" / str(condition["filename"]),
                delimiter=",",
                skiprows=1,
            )
        )
        count = len(values)
        names.extend([str(condition["name"])] * count)
        labels.extend([str(condition["label"])] * count)
        delays.extend([float(condition["delay_ms"])] * count)
        amplitudes.extend(values[:, 0].astype(float))
        changes.extend(values[:, 1].astype(float))
    return {
        "condition": np.asarray(names, dtype=object),
        "condition_label": np.asarray(labels, dtype=object),
        "delays_ms": np.asarray(delays, dtype=float),
        "initial_epsp_pa": np.asarray(amplitudes, dtype=float),
        "delta_weight_percent": np.asarray(changes, dtype=float),
    }


def solve_initial_weight_mapping(
    initial_epsp_pa: Sequence[float],
    model_initial_weight_start: float,
    model_initial_weight_range: float,
) -> tuple[float, float]:
    """Return the affine mapping from experimental pA to model units."""

    experimental = np.asarray(initial_epsp_pa, dtype=float)
    data_start = float(np.min(experimental))
    data_end = float(np.max(experimental))
    if data_end <= data_start:
        raise ValueError("initial EPSP amplitudes must span a non-zero range")
    if model_initial_weight_start < 0.0:
        raise ValueError("model_initial_weight_start must be non-negative")
    if model_initial_weight_range <= 0.0:
        raise ValueError("model_initial_weight_range must be positive")
    slope = model_initial_weight_range / (data_end - data_start)
    return float(slope), float(model_initial_weight_start - slope * data_start)


def _model_delta_weight_percent(
    delay_ms: float, initial_weight: float, params: Mapping[str, float]
) -> float:
    if initial_weight <= 0.0:
        raise ValueError("mapped model initial weights must be positive")
    change = stdp_single(
        delay_ms=float(delay_ms),
        transmission_delay_ms=float(params["transmission_delay_ms"]),
        decay_rate=float(params["decay_rate"]),
        plasticity_threshold=float(params["plasticity_threshold"]),
        learning_rate=float(params["learning_rate"]),
        initial_weight=float(initial_weight),
    )
    return 100.0 * float(np.asarray(change).squeeze()) / initial_weight


def evaluate_config(
    data: Mapping[str, Any], params: Mapping[str, float]
) -> dict[str, Any]:
    """Evaluate the unweighted sum-of-squared-error objective."""

    slope, intercept = solve_initial_weight_mapping(
        data["initial_epsp_pa"],
        float(params["model_initial_weight_start"]),
        float(params["model_initial_weight_range"]),
    )
    mapped = np.asarray(data["initial_epsp_pa"], dtype=float) * slope + intercept
    predicted = np.asarray(
        [
            _model_delta_weight_percent(delay, weight, params)
            for delay, weight in zip(data["delays_ms"], mapped)
        ],
        dtype=float,
    )
    observed = np.asarray(data["delta_weight_percent"], dtype=float)
    residuals = predicted - observed
    return {
        "objective": float(np.sum(np.square(residuals))),
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "mapping_slope_model_units_per_pa": slope,
        "mapping_intercept_model_units": intercept,
        "mapped_initial_weights": mapped,
        "predicted_percent": predicted,
    }


def initial_strength_fit_trainable(config: Mapping[str, Any]) -> dict[str, float]:
    """Pure Ray Tune/mini-radas trainable with no file writes."""

    source = config.get("parameters", config)
    if not isinstance(source, Mapping):
        raise TypeError("config['parameters'] must be a mapping when provided")
    params = {name: float(source[name]) for name in PARAMETER_NAMES}
    result = evaluate_config(load_dataset(), params)
    return {
        name: float(result[name])
        for name in (
            "objective",
            "rmse",
            "mapping_slope_model_units_per_pa",
            "mapping_intercept_model_units",
        )
    }


def search_bounds(search_space: str) -> dict[str, tuple[float, float, bool]]:
    """Return bounds as ``(low, high, logarithmic)`` triples."""

    if search_space not in {"legacy", "manuscript"}:
        raise ValueError("search_space must be 'legacy' or 'manuscript'")
    mapping_range = (0.5, 2.0, False) if search_space == "legacy" else (0.0, 0.3, False)
    return {
        "model_initial_weight_start": (0.0, 0.3, False),
        "model_initial_weight_range": mapping_range,
        "transmission_delay_ms": (0.0, 10.0, False),
        "decay_rate": (0.005, 0.5, True),
        "plasticity_threshold": (-0.5, 0.5, False),
        "learning_rate": (0.01, 1.0, True),
    }


def run_optuna_fit(
    data: Mapping[str, Any], num_samples: int, seed: int, search_space: str
) -> tuple[dict[str, float], Any]:
    """Run a serial local Optuna search (the mini-radas runner is preferred)."""

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Fresh fitting requires Optuna; install requirements-search.txt."
        ) from exc
    bounds = search_bounds(search_space)

    def objective(trial: Any) -> float:
        params = {
            name: trial.suggest_float(name, low, high, log=logarithmic)
            for name, (low, high, logarithmic) in bounds.items()
        }
        try:
            value = float(evaluate_config(data, params)["objective"])
        except (FloatingPointError, OverflowError, ValueError):
            return float("inf")
        return value if math.isfinite(value) else float("inf")

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=num_samples, n_jobs=1, show_progress_bar=False)
    return {name: float(study.best_params[name]) for name in PARAMETER_NAMES}, study


def _condition_curves(
    data: Mapping[str, Any], params: Mapping[str, float], points: int = 250
) -> dict[str, dict[str, np.ndarray]]:
    amplitudes = np.asarray(data["initial_epsp_pa"], dtype=float)
    slope, intercept = solve_initial_weight_mapping(
        amplitudes,
        float(params["model_initial_weight_start"]),
        float(params["model_initial_weight_range"]),
    )
    grid = np.logspace(np.log10(amplitudes.min()), np.log10(amplitudes.max()), points)
    mapped = grid * slope + intercept
    return {
        str(condition["name"]): {
            "initial_epsp_pa": grid,
            "mapped_initial_weight": mapped,
            "delta_weight_percent": np.asarray(
                [
                    _model_delta_weight_percent(
                        float(condition["delay_ms"]), weight, params
                    )
                    for weight in mapped
                ]
            ),
        }
        for condition in CONDITIONS
    }


def _write_csv(
    path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _plot(
    output: Path,
    data: Mapping[str, Any],
    curves: Mapping[str, Mapping[str, np.ndarray]],
    formats: Sequence[str],
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(3.45, 2.65), constrained_layout=True)
    axis.axhline(0.0, color="0.75", linewidth=0.7, zorder=0)
    styles = {
        "pre_post_5ms": ("#0072B2", "o", "-"),
        "post_pre_6ms": ("#D55E00", "x", "--"),
    }
    for condition in CONDITIONS:
        name = str(condition["name"])
        color, marker, line = styles[name]
        mask = np.asarray(data["condition"]) == name
        axis.scatter(
            np.asarray(data["initial_epsp_pa"])[mask],
            np.asarray(data["delta_weight_percent"])[mask],
            color=color,
            marker=marker,
            s=20,
            linewidths=0.9,
            label=f"{condition['label']} data",
            zorder=3,
        )
        curve = curves[name]
        axis.plot(
            curve["initial_epsp_pa"],
            curve["delta_weight_percent"],
            color=color,
            linestyle=line,
            linewidth=1.5,
            label=f"{condition['label']} PDM",
        )
    axis.set_xscale("log")
    axis.set_xlabel("Initial EPSP amplitude (pA)")
    axis.set_ylabel("Weight change (%)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=7)
    for suffix in formats:
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=dpi)
    plt.close(figure)


def materialize_result(
    data: Mapping[str, Any],
    params: Mapping[str, float],
    output: Path,
    formats: Sequence[str],
    dpi: int,
    source: str,
    search_space: str,
    seed: int | None = None,
    num_samples: int | None = None,
    study: Any | None = None,
) -> dict[str, Path]:
    """Write fitted values, curves, metadata, and plots."""

    output.mkdir(parents=True, exist_ok=True)
    evaluation = evaluate_config(data, params)
    curves = _condition_curves(data, params)
    stem = "fig7b_initial_strength"
    observations = output / f"{stem}_observations.csv"
    _write_csv(
        observations,
        (
            "condition",
            "delay_ms",
            "initial_epsp_pa",
            "mapped_model_initial_weight",
            "experimental_delta_weight_percent",
            "model_delta_weight_percent",
        ),
        list(
            zip(
                data["condition"],
                data["delays_ms"],
                data["initial_epsp_pa"],
                evaluation["mapped_initial_weights"],
                data["delta_weight_percent"],
                evaluation["predicted_percent"],
            )
        ),
    )
    curve_csv = output / f"{stem}_curve.csv"
    curve_rows = []
    for condition in CONDITIONS:
        name = str(condition["name"])
        curve = curves[name]
        curve_rows.extend(
            zip(
                [name] * len(curve["initial_epsp_pa"]),
                [condition["delay_ms"]] * len(curve["initial_epsp_pa"]),
                curve["initial_epsp_pa"],
                curve["mapped_initial_weight"],
                curve["delta_weight_percent"],
            )
        )
    _write_csv(
        curve_csv,
        (
            "condition",
            "delay_ms",
            "initial_epsp_pa",
            "mapped_model_initial_weight",
            "model_delta_weight_percent",
        ),
        curve_rows,
    )

    result_json = output / f"{stem}_fit.json"
    result_json.write_text(
        json.dumps(
            {
                "panel": "Fig. 7b",
                "dataset": "Bi and Poo (1998), Fig. 5",
                "source": source,
                "seed": seed,
                "num_samples": num_samples,
                "parameters": {name: float(params[name]) for name in PARAMETER_NAMES},
                "search_space_variant": search_space,
                "search_space": {
                    name: {
                        "distribution": "loguniform" if logarithmic else "uniform",
                        "low": low,
                        "high": high,
                    }
                    for name, (low, high, logarithmic) in search_bounds(
                        search_space
                    ).items()
                },
                "objective": "unweighted sum((model_percent - observed_percent)^2)",
                "objective_value": evaluation["objective"],
                "rmse": evaluation["rmse"],
                "n_observations": int(len(data["initial_epsp_pa"])),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _plot(output / stem, data, curves, formats, dpi)
    paths = {
        "json": result_json,
        "observations_csv": observations,
        "curve_csv": curve_csv,
    }
    if study is not None:
        trials = output / f"{stem}_trials.csv"
        _write_csv(
            trials,
            ("trial", "state", "objective", *PARAMETER_NAMES),
            [
                (
                    trial.number,
                    trial.state.name,
                    "" if trial.value is None else trial.value,
                    *(trial.params.get(name, "") for name in PARAMETER_NAMES),
                )
                for trial in study.trials
            ],
        )
        paths["trials_csv"] = trials
    for suffix in formats:
        paths[suffix] = (output / stem).with_suffix(f".{suffix}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit", action="store_true", help="run a fresh local Optuna fit"
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--search-space", choices=("legacy", "manuscript"), default="legacy"
    )
    parser.add_argument(
        "--manuscript-search-space",
        dest="search_space",
        action="store_const",
        const="manuscript",
    )
    parser.add_argument("--output", type=Path, default=HERE / "generated" / "fig7b_fit")
    parser.add_argument(
        "--formats", nargs="+", choices=("pdf", "svg"), default=["pdf", "svg"]
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.fit and args.search_space != "legacy":
        parser.error(
            "the embedded configuration was obtained from the legacy search space"
        )
    data = load_dataset()
    if args.fit:
        params, study = run_optuna_fit(
            data, args.num_samples, args.seed, args.search_space
        )
        source = "fresh_optuna"
    else:
        params, study = EMBEDDED_BEST_CONFIG.copy(), None
        source = "migrated_notebook_best_config"
    paths = materialize_result(
        data,
        params,
        args.output,
        args.formats,
        args.dpi,
        source,
        args.search_space,
        args.seed if args.fit else None,
        args.num_samples if args.fit else None,
        study,
    )
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
