"""Reproduce or refit the experimental STDP curves in Fig. 6i-j.

The default path uses the best configurations recorded by the original
Optuna notebooks and only evaluates the model, so it is quick and has no
dependency on Ray, mini-radas, or remote experiment checkpoints.  Passing
``--fit`` starts a new, local Optuna study.
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

# Support both ``python spiking/fit_stdp.py`` and
# ``python -m spiking.fit_stdp`` without depending on an installed package.
if __package__:
    from .model import stdp_single
else:  # pragma: no cover - the CLI smoke test covers this import route
    sys.path.insert(0, str(HERE))
    from model import stdp_single


DATASET_SPECS: dict[str, dict[str, str]] = {
    "bi2002": {
        "filename": "Bi et al. 2002.csv",
        "label": "Bi and Wang (2002)",
        "panel": "Fig. 6i",
        "stem": "fig6i_bi2002",
        "transform": "(digitized_y - 1) * 100",
    },
    "woodin2003": {
        "filename": "Melanie et al. 2003.csv",
        "label": "Woodin et al. (2003)",
        "panel": "Fig. 6j",
        "stem": "fig6j_woodin2003",
        "transform": "none",
    },
}

# Exact best configurations printed by the migrated OptunaSearch notebooks.
# These values were optimized with the legacy observation-index variance
# weighting implemented in ``legacy_variance_weights`` below.
EMBEDDED_BEST_CONFIGS: dict[str, dict[str, float]] = {
    "bi2002": {
        "transmission_delay_ms": 3.003219049874547,
        "decay_rate": 0.09832289732846322,
        "plasticity_threshold": 0.2289279077211322,
        "learning_rate": 0.3412298607505473,
        "initial_weight": 0.7286125771004283,
    },
    "woodin2003": {
        "transmission_delay_ms": 4.992491557428629,
        "decay_rate": 0.024879461150547056,
        "plasticity_threshold": -0.0686892325661495,
        "learning_rate": 0.4564539181420359,
        "initial_weight": 0.4313814751057967,
    },
}

PARAMETER_NAMES = tuple(next(iter(EMBEDDED_BEST_CONFIGS.values())).keys())


def load_dataset(dataset: str) -> dict[str, Any]:
    """Load one digitized dataset and apply its documented y transform."""

    if dataset not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset {dataset!r}; choose from {DATASET_SPECS}")
    spec = DATASET_SPECS[dataset]
    values = np.atleast_2d(
        np.loadtxt(HERE / "data" / spec["filename"], delimiter=",", skiprows=1)
    )
    delays_ms = values[:, 0].astype(float, copy=True)
    delta_weight_percent = values[:, 1].astype(float, copy=True)
    if dataset == "bi2002":
        delta_weight_percent = (delta_weight_percent - 1.0) * 100.0
    return {
        "dataset": dataset,
        "delays_ms": delays_ms,
        "delta_weight_percent": delta_weight_percent,
        **spec,
    }


def legacy_variance_weights(
    delays_ms: Sequence[float],
    delta_weight_percent: Sequence[float],
    half_window_observations: int = 5,
) -> np.ndarray:
    """Reproduce the original notebook's index-window weighting exactly.

    Despite its name, the historical window was measured in sorted
    observations rather than milliseconds.  ``list.index`` also caused every
    duplicate delay to use the window around its first occurrence.  Both
    details are intentionally retained for numerical provenance.
    """

    if half_window_observations < 1:
        raise ValueError("half_window_observations must be at least 1")
    xs = [float(value) for value in delays_ms]
    ys = [float(value) for value in delta_weight_percent]
    paired_sorted = sorted(zip(xs, ys))
    xs_sorted, ys_reordered = zip(*paired_sorted)
    weights: list[float] = []
    for delay in xs:
        index = xs_sorted.index(delay)
        start = max(0, index - half_window_observations)
        # This exclusive end is the historical implementation: at most ten
        # observations for half_window_observations=5, not eleven.
        end = min(len(xs_sorted), index + half_window_observations)
        variance = float(np.var(ys_reordered[start:end]))
        if variance == 0.0 or not math.isfinite(variance):
            raise ValueError("legacy variance window produced a non-positive variance")
        weights.append(1.0 / variance)
    return np.asarray(weights, dtype=float)


def manuscript_variance_weights(
    delays_ms: Sequence[float],
    delta_weight_percent: Sequence[float],
    half_width_ms: float = 5.0,
    variance_floor: float = 1e-12,
) -> np.ndarray:
    """Weight by reciprocal variance inside the manuscript's +/-5 ms window.

    The paper specifies an open interval.  Some digitized points have no
    non-identical neighbour in that interval and therefore have zero local
    variance.  A declared floor keeps the objective finite; it is recorded in
    every output JSON file.
    """

    if half_width_ms <= 0:
        raise ValueError("half_width_ms must be positive")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")
    xs = np.asarray(delays_ms, dtype=float)
    ys = np.asarray(delta_weight_percent, dtype=float)
    weights = np.empty_like(xs)
    for index, delay in enumerate(xs):
        in_window = (xs > delay - half_width_ms) & (xs < delay + half_width_ms)
        variance = max(float(np.var(ys[in_window])), variance_floor)
        weights[index] = 1.0 / variance
    return weights


def _model_delta_weight_percent(delay_ms: float, params: Mapping[str, float]) -> float:
    """Adapt the model's absolute weight change to percent of initial weight."""

    initial_weight = float(params["initial_weight"])
    if initial_weight <= 0.0:
        raise ValueError("initial_weight must be positive for percentage change")
    result = stdp_single(
        delay_ms=float(delay_ms),
        transmission_delay_ms=float(params["transmission_delay_ms"]),
        decay_rate=float(params["decay_rate"]),
        plasticity_threshold=float(params["plasticity_threshold"]),
        learning_rate=float(params["learning_rate"]),
        initial_weight=initial_weight,
    )
    # The local publication model returns a scalar.  Accepting the first item
    # of the old (delta_w, traces...) convention makes migration failures much
    # easier to diagnose without introducing an old-model dependency.
    if isinstance(result, (tuple, list)):
        result = result[0]
    raw_change = float(np.asarray(result).squeeze())
    return 100.0 * raw_change / initial_weight


def variance_weights(
    data: Mapping[str, Any], weighting: str, variance_floor: float = 1e-12
) -> np.ndarray:
    delays = data["delays_ms"]
    changes = data["delta_weight_percent"]
    if weighting == "legacy":
        return legacy_variance_weights(delays, changes)
    if weighting == "manuscript":
        return manuscript_variance_weights(
            delays, changes, half_width_ms=5.0, variance_floor=variance_floor
        )
    if weighting == "none":
        return np.ones(len(delays), dtype=float)
    raise ValueError("weighting must be 'legacy', 'manuscript', or 'none'")


def evaluate_config(
    data: Mapping[str, Any],
    params: Mapping[str, float],
    weighting: str = "legacy",
    variance_floor: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate the historical normalized, variance-weighted SSE objective."""

    delays = np.asarray(data["delays_ms"], dtype=float)
    observed = np.asarray(data["delta_weight_percent"], dtype=float)
    unscaled = np.asarray(
        [_model_delta_weight_percent(delay, params) for delay in delays], dtype=float
    )
    model_sum = float(np.sum(unscaled))
    observed_sum = float(np.sum(observed))
    if model_sum == 0.0 or not math.isfinite(model_sum):
        raise FloatingPointError("model predictions cannot be normalized: invalid sum")
    scale = observed_sum / model_sum
    predicted = unscaled * scale
    weights = variance_weights(data, weighting, variance_floor)
    residuals = predicted - observed
    objective = float(np.sum(np.square(residuals) * weights))
    return {
        "objective": objective,
        "unweighted_sse": float(np.sum(np.square(residuals))),
        "unweighted_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "normalization_scale": float(scale),
        "predicted_percent": predicted,
        "variance_weights": weights,
    }


def stdp_fit_trainable(config: Mapping[str, Any]) -> dict[str, float]:
    """Pure trainable for optional Ray Tune/mini-radas orchestration.

    Parameters may be flat (the natural Tune representation) or nested under
    ``parameters``. Infrastructure-only fields are ignored and neither
    mini-radas nor Ray is imported here.
    """

    dataset = str(config.get("dataset", "bi2002"))
    weighting = str(config.get("weighting", "legacy"))
    variance_floor = float(config.get("variance_floor", 1e-12))
    parameter_source = config.get("parameters", config)
    if not isinstance(parameter_source, Mapping):
        raise TypeError("config['parameters'] must be a mapping when provided")
    params = {name: float(parameter_source[name]) for name in PARAMETER_NAMES}
    evaluation = evaluate_config(
        load_dataset(dataset),
        params,
        weighting=weighting,
        variance_floor=variance_floor,
    )
    return {
        "objective": float(evaluation["objective"]),
        "unweighted_sse": float(evaluation["unweighted_sse"]),
        "unweighted_rmse": float(evaluation["unweighted_rmse"]),
    }


def _objective_for_optuna(
    trial: Any,
    data: Mapping[str, Any],
    weighting: str,
    variance_floor: float,
) -> float:
    params = {
        "transmission_delay_ms": trial.suggest_float(
            "transmission_delay_ms", 0.0, 10.0
        ),
        "decay_rate": trial.suggest_float("decay_rate", 0.005, 0.5, log=True),
        "plasticity_threshold": trial.suggest_float("plasticity_threshold", -0.5, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "initial_weight": trial.suggest_float("initial_weight", 0.0, 2.0),
    }
    try:
        objective = evaluate_config(data, params, weighting, variance_floor)[
            "objective"
        ]
    except (FloatingPointError, OverflowError, ValueError):
        return float("inf")
    return objective if math.isfinite(objective) else float("inf")


def run_optuna_fit(
    data: Mapping[str, Any],
    weighting: str,
    variance_floor: float,
    num_samples: int,
    seed: int,
) -> tuple[dict[str, float], Any]:
    """Run a fresh local Optuna study and return its best parameters."""

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
        lambda trial: _objective_for_optuna(
            trial, data=data, weighting=weighting, variance_floor=variance_floor
        ),
        n_trials=num_samples,
        n_jobs=1,
        show_progress_bar=False,
    )
    return {name: float(study.best_params[name]) for name in PARAMETER_NAMES}, study


def _curve(
    data: Mapping[str, Any], params: Mapping[str, float], scale: float
) -> tuple[np.ndarray, np.ndarray]:
    # Match the one-millisecond grid used by the original plotting notebook.
    delays = np.arange(
        float(np.min(data["delays_ms"])),
        float(np.max(data["delays_ms"])),
        1.0,
    )
    changes = np.asarray(
        [_model_delta_weight_percent(delay, params) for delay in delays], dtype=float
    )
    return delays, changes * scale


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
    curve_delays: np.ndarray,
    curve_changes: np.ndarray,
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
    ax.scatter(
        data["delays_ms"],
        data["delta_weight_percent"],
        s=19,
        facecolors="#0072B2",
        edgecolors="#0072B2",
        linewidths=0.9,
        label="Data",
        zorder=3,
    )
    ax.plot(
        curve_delays,
        curve_changes,
        color="#D55E00",
        linewidth=1.5,
        label="PDM",
        zorder=2,
    )
    ax.set_xlabel("Pre-post interval (ms)")
    ax.set_ylabel("Weight change (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, handlelength=1.8)
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
    weighting: str,
    variance_floor: float,
    output: Path,
    formats: Sequence[str],
    dpi: int,
    source: str,
    seed: int | None = None,
    num_samples: int | None = None,
    study: Any | None = None,
) -> dict[str, Path]:
    """Write numerical results, provenance, and vector figure files."""

    output.mkdir(parents=True, exist_ok=True)
    stem = str(data["stem"])
    evaluation = evaluate_config(data, params, weighting, variance_floor)
    curve_delays, curve_changes = _curve(
        data, params, evaluation["normalization_scale"]
    )

    observed_csv = output / f"{stem}_observations.csv"
    _write_csv(
        observed_csv,
        [
            "delay_ms",
            "experimental_delta_weight_percent",
            "model_delta_weight_percent",
            "objective_weight",
        ],
        list(
            zip(
                data["delays_ms"],
                data["delta_weight_percent"],
                evaluation["predicted_percent"],
                evaluation["variance_weights"],
            )
        ),
    )
    curve_csv = output / f"{stem}_curve.csv"
    _write_csv(
        curve_csv,
        ["delay_ms", "model_delta_weight_percent"],
        list(zip(curve_delays, curve_changes)),
    )

    result_json = output / f"{stem}_fit.json"
    payload = {
        "panel": data["panel"],
        "dataset": data["dataset"],
        "dataset_label": data["label"],
        "source": source,
        "seed": seed,
        "num_samples": num_samples,
        "parameters": {name: float(params[name]) for name in PARAMETER_NAMES},
        "search_space": {
            "transmission_delay_ms": {
                "distribution": "uniform",
                "low": 0.0,
                "high": 10.0,
            },
            "decay_rate": {"distribution": "loguniform", "low": 0.005, "high": 0.5},
            "plasticity_threshold": {
                "distribution": "uniform",
                "low": -0.5,
                "high": 0.5,
            },
            "learning_rate": {"distribution": "loguniform", "low": 0.01, "high": 1.0},
            "initial_weight": {"distribution": "uniform", "low": 0.0, "high": 2.0},
        },
        "data_transform": data["transform"],
        "objective": "sum(weight * (normalized_model_percent - observed_percent)^2)",
        "weighting": weighting,
        "variance_floor": variance_floor if weighting == "manuscript" else None,
        "normalization": "model predictions scaled so their sum at observed delays equals the observed-data sum",
        "normalization_scale": evaluation["normalization_scale"],
        "objective_value": evaluation["objective"],
        "unweighted_sse": evaluation["unweighted_sse"],
        "unweighted_rmse": evaluation["unweighted_rmse"],
        "n_observations": int(len(data["delays_ms"])),
    }
    result_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    figure_base = output / stem
    _plot(figure_base, data, curve_delays, curve_changes, formats, dpi)
    paths = {
        "json": result_json,
        "observations_csv": observed_csv,
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
        "--dataset",
        choices=[*DATASET_SPECS, "all"],
        default="all",
        help="dataset/panel to reproduce (default: both)",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="run a fresh Optuna fit instead of using the embedded best configuration",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Optuna trials per selected dataset when --fit is used (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Optuna sampler seed")
    parser.add_argument(
        "--weighting",
        choices=["legacy", "manuscript", "none"],
        default="legacy",
        help="variance weighting definition (default: exact legacy notebook behavior)",
    )
    parser.add_argument(
        "--manuscript-weighting",
        dest="weighting",
        action="store_const",
        const="manuscript",
        help="shortcut for --weighting manuscript (strict +/-5 ms windows)",
    )
    parser.add_argument(
        "--variance-floor",
        type=float,
        default=1e-12,
        help="positive floor for zero-variance manuscript windows",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "generated" / "fig6_fits",
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
    args = build_parser().parse_args(argv)
    selected = list(DATASET_SPECS) if args.dataset == "all" else [args.dataset]
    for dataset_index, dataset in enumerate(selected):
        data = load_dataset(dataset)
        if args.fit:
            effective_seed = args.seed + dataset_index
            params, study = run_optuna_fit(
                data=data,
                weighting=args.weighting,
                variance_floor=args.variance_floor,
                num_samples=args.num_samples,
                seed=effective_seed,
            )
            source = "fresh_optuna"
        else:
            effective_seed = None
            params = EMBEDDED_BEST_CONFIGS[dataset].copy()
            study = None
            source = "migrated_notebook_best_config"
        paths = materialize_result(
            data=data,
            params=params,
            weighting=args.weighting,
            variance_floor=args.variance_floor,
            output=args.output,
            formats=args.formats,
            dpi=args.dpi,
            source=source,
            seed=effective_seed,
            num_samples=args.num_samples if args.fit else None,
            study=study,
        )
        print(f"{data['panel']} ({data['label']}):")
        for kind, path in paths.items():
            print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
