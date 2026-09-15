"""Reproduce or refit the initial-strength dependence in Fig. 7b.

By default this script evaluates the best configuration recorded by the
original Optuna notebook.  Use ``--fit`` to launch a fresh, local Optuna
study; no external experiment checkpoint is required.
"""

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
else:  # pragma: no cover - exercised by the CLI rather than module tests
    sys.path.insert(0, str(HERE))
    from model import stdp_single


# Exact best configuration printed by fit_stdp_data_initial_w.ipynb.
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
    """Load the two digitized Bi and Poo (1998) conditions."""

    condition_names: list[str] = []
    condition_labels: list[str] = []
    delays_ms: list[float] = []
    initial_epsp_pa: list[float] = []
    delta_weight_percent: list[float] = []
    for condition in CONDITIONS:
        values = np.atleast_2d(
            np.loadtxt(
                HERE / "data" / str(condition["filename"]),
                delimiter=",",
                skiprows=1,
            )
        )
        count = len(values)
        condition_names.extend([str(condition["name"])] * count)
        condition_labels.extend([str(condition["label"])] * count)
        delays_ms.extend([float(condition["delay_ms"])] * count)
        initial_epsp_pa.extend(values[:, 0].astype(float))
        delta_weight_percent.extend(values[:, 1].astype(float))
    return {
        "condition": np.asarray(condition_names, dtype=object),
        "condition_label": np.asarray(condition_labels, dtype=object),
        "delays_ms": np.asarray(delays_ms, dtype=float),
        "initial_epsp_pa": np.asarray(initial_epsp_pa, dtype=float),
        "delta_weight_percent": np.asarray(delta_weight_percent, dtype=float),
    }


def solve_initial_weight_mapping(
    initial_epsp_pa: Sequence[float],
    model_initial_weight_start: float,
    model_initial_weight_range: float,
) -> tuple[float, float]:
    """Return slope and intercept mapping experimental pA to model units."""

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
    intercept = model_initial_weight_start - slope * data_start
    return float(slope), float(intercept)


def _model_delta_weight_percent(
    delay_ms: float, initial_weight: float, params: Mapping[str, float]
) -> float:
    if initial_weight <= 0.0:
        raise ValueError("mapped model initial weights must be positive")
    result = stdp_single(
        delay_ms=float(delay_ms),
        transmission_delay_ms=float(params["transmission_delay_ms"]),
        decay_rate=float(params["decay_rate"]),
        plasticity_threshold=float(params["plasticity_threshold"]),
        learning_rate=float(params["learning_rate"]),
        initial_weight=float(initial_weight),
    )
    if isinstance(result, (tuple, list)):
        result = result[0]
    absolute_change = float(np.asarray(result).squeeze())
    return 100.0 * absolute_change / initial_weight


def evaluate_config(
    data: Mapping[str, Any], params: Mapping[str, float]
) -> dict[str, Any]:
    """Evaluate the unweighted sum-of-squared-error objective from the paper."""

    slope, intercept = solve_initial_weight_mapping(
        data["initial_epsp_pa"],
        float(params["model_initial_weight_start"]),
        float(params["model_initial_weight_range"]),
    )
    mapped_weights = (
        np.asarray(data["initial_epsp_pa"], dtype=float) * slope + intercept
    )
    predicted = np.asarray(
        [
            _model_delta_weight_percent(delay, initial_weight, params)
            for delay, initial_weight in zip(data["delays_ms"], mapped_weights)
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
        "mapped_initial_weights": mapped_weights,
        "predicted_percent": predicted,
    }


def initial_strength_fit_trainable(config: Mapping[str, Any]) -> dict[str, float]:
    """Pure trainable for optional Ray Tune/mini-radas orchestration.

    The parameter mapping may be supplied flat or under ``parameters``. This
    function imports neither Ray nor mini-radas and performs no file writes.
    """

    parameter_source = config.get("parameters", config)
    if not isinstance(parameter_source, Mapping):
        raise TypeError("config['parameters'] must be a mapping when provided")
    params = {name: float(parameter_source[name]) for name in PARAMETER_NAMES}
    evaluation = evaluate_config(load_dataset(), params)
    return {
        "objective": float(evaluation["objective"]),
        "rmse": float(evaluation["rmse"]),
        "mapping_slope_model_units_per_pa": float(
            evaluation["mapping_slope_model_units_per_pa"]
        ),
        "mapping_intercept_model_units": float(
            evaluation["mapping_intercept_model_units"]
        ),
    }


def _search_bounds(search_space: str) -> dict[str, tuple[float, float, bool]]:
    if search_space not in {"legacy", "manuscript"}:
        raise ValueError("search_space must be 'legacy' or 'manuscript'")
    # The migrated notebook used [0.5, 2.0] for model_initial_weight_range.
    # Methods Section H states [0.0, 0.3], which cannot contain the reported
    # best value 0.651577...; expose both and retain the runnable provenance as
    # the default.
    mapping_range = (0.5, 2.0, False) if search_space == "legacy" else (0.0, 0.3, False)
    return {
        "model_initial_weight_start": (0.0, 0.3, False),
        "model_initial_weight_range": mapping_range,
        "transmission_delay_ms": (0.0, 10.0, False),
        "decay_rate": (0.005, 0.5, True),
        "plasticity_threshold": (-0.5, 0.5, False),
        "learning_rate": (0.01, 1.0, True),
    }


def _objective_for_optuna(
    trial: Any, data: Mapping[str, Any], search_space: str
) -> float:
    bounds = _search_bounds(search_space)
    params = {
        name: trial.suggest_float(name, low, high, log=log)
        for name, (low, high, log) in bounds.items()
    }
    try:
        objective = evaluate_config(data, params)["objective"]
    except (FloatingPointError, OverflowError, ValueError):
        return float("inf")
    return objective if math.isfinite(objective) else float("inf")


def run_optuna_fit(
    data: Mapping[str, Any], num_samples: int, seed: int, search_space: str
) -> tuple[dict[str, float], Any]:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Fresh fitting requires Optuna; install the spiking requirements first."
        ) from exc
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: _objective_for_optuna(trial, data, search_space),
        n_trials=num_samples,
        n_jobs=1,
        show_progress_bar=False,
    )
    return {name: float(study.best_params[name]) for name in PARAMETER_NAMES}, study


def _condition_curves(
    data: Mapping[str, Any], params: Mapping[str, float], points: int = 250
) -> dict[str, dict[str, np.ndarray]]:
    initial_epsp = np.asarray(data["initial_epsp_pa"], dtype=float)
    slope, intercept = solve_initial_weight_mapping(
        initial_epsp,
        float(params["model_initial_weight_start"]),
        float(params["model_initial_weight_range"]),
    )
    grid = np.logspace(
        np.log10(initial_epsp.min()), np.log10(initial_epsp.max()), points
    )
    mapped = grid * slope + intercept
    curves: dict[str, dict[str, np.ndarray]] = {}
    for condition in CONDITIONS:
        delay = float(condition["delay_ms"])
        changes = np.asarray(
            [_model_delta_weight_percent(delay, weight, params) for weight in mapped],
            dtype=float,
        )
        curves[str(condition["name"])] = {
            "initial_epsp_pa": grid,
            "mapped_initial_weight": mapped,
            "delta_weight_percent": changes,
        }
    return curves


def _write_csv(
    path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_trials(path: Path, study: Any) -> None:
    rows = []
    for trial in study.trials:
        rows.append(
            [
                trial.number,
                trial.state.name,
                "" if trial.value is None else trial.value,
                *[trial.params.get(name, "") for name in PARAMETER_NAMES],
            ]
        )
    _write_csv(path, ["trial", "state", "objective", *PARAMETER_NAMES], rows)


def _plot(
    path_without_suffix: Path,
    data: Mapping[str, Any],
    curves: Mapping[str, Mapping[str, np.ndarray]],
    formats: Sequence[str],
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.65), constrained_layout=True)
    ax.axhline(0.0, color="0.75", linewidth=0.7, zorder=0)
    styles = {
        "pre_post_5ms": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
        "post_pre_6ms": {"color": "#D55E00", "marker": "x", "linestyle": "--"},
    }
    for condition in CONDITIONS:
        name = str(condition["name"])
        mask = np.asarray(data["condition"]) == name
        style = styles[name]
        scatter_kwargs: dict[str, Any] = {
            "s": 20,
            "marker": style["marker"],
            "linewidths": 0.9,
            "label": f"{condition['label']} data",
            "zorder": 3,
        }
        scatter_kwargs.update(color=style["color"])
        ax.scatter(
            np.asarray(data["initial_epsp_pa"])[mask],
            np.asarray(data["delta_weight_percent"])[mask],
            **scatter_kwargs,
        )
        curve = curves[name]
        ax.plot(
            curve["initial_epsp_pa"],
            curve["delta_weight_percent"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            label=f"{condition['label']} PDM",
            zorder=2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Initial EPSP amplitude (pA)")
    ax.set_ylabel("Weight change (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, handlelength=2.0)
    for suffix in formats:
        fig.savefig(
            path_without_suffix.with_suffix(f".{suffix}"),
            dpi=dpi,
            metadata={"Creator": "PDM reproduction scripts"},
        )
    plt.close(fig)


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
    output.mkdir(parents=True, exist_ok=True)
    evaluation = evaluate_config(data, params)
    curves = _condition_curves(data, params)
    stem = "fig7b_initial_strength"

    observations_csv = output / f"{stem}_observations.csv"
    _write_csv(
        observations_csv,
        [
            "condition",
            "delay_ms",
            "initial_epsp_pa",
            "mapped_model_initial_weight",
            "experimental_delta_weight_percent",
            "model_delta_weight_percent",
        ],
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
        [
            "condition",
            "delay_ms",
            "initial_epsp_pa",
            "mapped_model_initial_weight",
            "model_delta_weight_percent",
        ],
        curve_rows,
    )

    bounds = _search_bounds(search_space)
    result_json = output / f"{stem}_fit.json"
    payload = {
        "panel": "Fig. 7b",
        "dataset": "Bi and Poo (1998), Fig. 5",
        "source": source,
        "seed": seed,
        "num_samples": num_samples,
        "parameters": {name: float(params[name]) for name in PARAMETER_NAMES},
        "search_space_variant": search_space,
        "search_space": {
            name: {
                "distribution": "loguniform" if log else "uniform",
                "low": low,
                "high": high,
            }
            for name, (low, high, log) in bounds.items()
        },
        "objective": "unweighted sum((model_percent - observed_percent)^2)",
        "objective_value": evaluation["objective"],
        "rmse": evaluation["rmse"],
        "mapping_slope_model_units_per_pa": evaluation[
            "mapping_slope_model_units_per_pa"
        ],
        "mapping_intercept_model_units": evaluation["mapping_intercept_model_units"],
        "n_observations": int(len(data["initial_epsp_pa"])),
        "conditions": {
            "positive_spiking.csv": "pre before post by 5 ms",
            "negative_spiking.csv": "post before pre by 6 ms",
        },
    }
    result_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    figure_base = output / stem
    _plot(figure_base, data, curves, formats, dpi)
    paths = {
        "json": result_json,
        "observations_csv": observations_csv,
        "curve_csv": curve_csv,
    }
    if study is not None:
        trials_csv = output / f"{stem}_trials.csv"
        _write_trials(trials_csv, study)
        paths["trials_csv"] = trials_csv
    for suffix in formats:
        paths[suffix] = figure_base.with_suffix(f".{suffix}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit",
        action="store_true",
        help="run a fresh Optuna fit instead of the embedded best configuration",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="number of Optuna trials when --fit is used (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Optuna sampler seed")
    parser.add_argument(
        "--search-space",
        choices=["legacy", "manuscript"],
        default="legacy",
        help=(
            "mapping-range bounds for --fit "
            "(default: bounds used by the migrated notebook)"
        ),
    )
    parser.add_argument(
        "--manuscript-search-space",
        dest="search_space",
        action="store_const",
        const="manuscript",
        help="shortcut for the mapping-range bounds printed in Methods Section H",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "generated" / "fig7b_fit",
        help="output directory for JSON, CSV, PDF, and SVG files",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["pdf", "svg"],
        default=["pdf", "svg"],
        help="figure formats to write",
    )
    parser.add_argument("--dpi", type=int, default=300, help="rasterization DPI")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.fit and args.search_space != "legacy":
        parser.error(
            "--search-space manuscript requires --fit; the embedded best "
            "configuration was obtained from the legacy search space"
        )
    data = load_dataset()
    if args.fit:
        params, study = run_optuna_fit(
            data=data,
            num_samples=args.num_samples,
            seed=args.seed,
            search_space=args.search_space,
        )
        source = "fresh_optuna"
    else:
        params = EMBEDDED_BEST_CONFIG.copy()
        study = None
        source = "migrated_notebook_best_config"
    paths = materialize_result(
        data=data,
        params=params,
        output=args.output,
        formats=args.formats,
        dpi=args.dpi,
        source=source,
        search_space=args.search_space,
        seed=args.seed if args.fit else None,
        num_samples=args.num_samples if args.fit else None,
        study=study,
    )
    print("Fig. 7b (Bi and Poo, 1998):")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
