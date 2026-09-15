"""Generate and plot the core numerical panels of manuscript Figure 6.

Panels a and c contain state traces from the historically omitted
``stdp_dx`` notebook. Panels b and d--g are simulated STDP curves, and panel h
compares the sigmoid-gated simulation with the step-gate approximation from
Eq. 37. Generation writes a portable long-form CSV; plotting only reads it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .analytic import analytic_stdp_curve
    from .model import SimulationResult, stdp_curve, stdp_single
except ImportError:  # Support ``python figure6.py`` from any directory.
    from analytic import analytic_stdp_curve
    from model import SimulationResult, stdp_curve, stdp_single


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results" / "figure6.csv"
DEFAULT_METADATA = HERE / "results" / "figure6.json"
DEFAULT_PLOT_DIR = HERE / "plots"

CSV_FIELDS = [
    "panel",
    "series",
    "variable",
    "time_ms",
    "delay_ms",
    "value",
    "lambda_per_ms",
    "tau_ms",
    "gamma",
    "alpha",
    "initial_weight",
]
TRACE_VARIABLES = (
    "input_signal",
    "dendritic_potential",
    "output_signal",
    "dendritic_error",
    "weight",
)
TRACE_LABELS = {
    "input_signal": r"$x^d$",
    "dendritic_potential": r"$v^d$",
    "output_signal": r"$x^{out}$",
    "dendritic_error": r"$\varepsilon^d$",
    "weight": r"$w^d$",
}
PANEL_TITLES = {
    "6b": "",
    "6d": "Membrane time constant",
    "6e": "Learning rate",
    "6f": "Transmission delay",
    "6g": "Activity required for plasticity",
    "6h": "",
}
PANEL_PALETTES = {
    "6b": ["#4c72b0"],
    "6d": ["#b7ddb0", "#6fba75", "#287a45"],
    "6e": ["#fcbba1", "#fb6a4a", "#cb181d"],
    "6f": ["#bdd7e7", "#6baed6", "#2171b5"],
    "6g": ["#dadaeb", "#bcbddc", "#807dba", "#54278f"],
    "6h": ["#4c72b0", "#c44e52"],
}


def _parameters(
    *,
    decay_rate: float,
    transmission_delay_ms: float,
    plasticity_threshold: float,
    learning_rate: float,
    initial_weight: float,
) -> dict[str, float]:
    return {
        "decay_rate": float(decay_rate),
        "transmission_delay_ms": float(transmission_delay_ms),
        "plasticity_threshold": float(plasticity_threshold),
        "learning_rate": float(learning_rate),
        "initial_weight": float(initial_weight),
    }


def panel_data(
    case: str,
    config: Mapping[str, object] | None = None,
) -> dict[str, list[float]]:
    """Pure serializable dispatcher suitable for local or mini-radas execution.

    Supported cases are ``"stdp_curve"``, ``"stdp_trace"`` and
    ``"analytic_curve"``. The function performs no file or plotting I/O and
    returns only lists of built-in floats.
    """

    values = dict(config or {})
    if case == "stdp_curve":
        delays = np.asarray(
            values.pop("delays_ms", np.arange(-50.0, 50.0)), dtype=float
        )
        curve = stdp_curve(delays, percent_change=True, **values)
        return {
            "delay_ms": delays.astype(float).tolist(),
            "weight_change_percent": curve.astype(float).tolist(),
        }
    if case == "analytic_curve":
        delays = np.asarray(
            values.pop("delays_ms", np.arange(-50.0, 50.0)), dtype=float
        )
        curve = analytic_stdp_curve(delays, percent_change=True, **values)
        return {
            "delay_ms": delays.astype(float).tolist(),
            "weight_change_percent": curve.astype(float).tolist(),
        }
    if case == "stdp_trace":
        delay_ms = float(values.pop("delay_ms"))
        _, simulation = stdp_single(delay_ms, return_trace=True, **values)
        return {
            "time_ms": simulation.time_ms.astype(float).tolist(),
            "input_signal": simulation.input_signal[0].astype(float).tolist(),
            "dendritic_potential": simulation.dendritic_potential[0]
            .astype(float)
            .tolist(),
            "output_signal": simulation.output_signal.astype(float).tolist(),
            "dendritic_error": simulation.dendritic_error[0].astype(float).tolist(),
            "weight": simulation.weights[0].astype(float).tolist(),
        }
    raise ValueError(
        "unknown case; expected 'stdp_curve', 'stdp_trace', or 'analytic_curve'"
    )


def _curve_rows(
    *,
    panel: str,
    series: str,
    delays_ms: np.ndarray,
    parameters: dict[str, float],
    dt_ms: float,
    gate_gain: float,
) -> list[dict[str, object]]:
    changes = stdp_curve(
        delays_ms,
        transmission_delay_ms=parameters["transmission_delay_ms"],
        decay_rate=parameters["decay_rate"],
        plasticity_threshold=parameters["plasticity_threshold"],
        learning_rate=parameters["learning_rate"],
        initial_weight=parameters["initial_weight"],
        dt_ms=dt_ms,
        gate_gain=gate_gain,
        percent_change=True,
    )
    return [
        {
            "panel": panel,
            "series": series,
            "variable": "weight_change_percent",
            "time_ms": "",
            "delay_ms": float(delay_ms),
            "value": float(change),
            "lambda_per_ms": parameters["decay_rate"],
            "tau_ms": parameters["transmission_delay_ms"],
            "gamma": parameters["plasticity_threshold"],
            "alpha": parameters["learning_rate"],
            "initial_weight": parameters["initial_weight"],
        }
        for delay_ms, change in zip(delays_ms, changes)
    ]


def _analytic_rows(
    *,
    delays_ms: np.ndarray,
    parameters: dict[str, float],
) -> list[dict[str, object]]:
    changes = analytic_stdp_curve(
        delays_ms,
        transmission_delay_ms=parameters["transmission_delay_ms"],
        decay_rate=parameters["decay_rate"],
        plasticity_threshold=parameters["plasticity_threshold"],
        learning_rate=parameters["learning_rate"],
        initial_weight=parameters["initial_weight"],
        percent_change=True,
    )
    return [
        {
            "panel": "6h",
            "series": "Approximate analytic",
            "variable": "weight_change_percent",
            "time_ms": "",
            "delay_ms": float(delay_ms),
            "value": float(change),
            "lambda_per_ms": parameters["decay_rate"],
            "tau_ms": parameters["transmission_delay_ms"],
            "gamma": parameters["plasticity_threshold"],
            "alpha": parameters["learning_rate"],
            "initial_weight": parameters["initial_weight"],
        }
        for delay_ms, change in zip(delays_ms, changes)
    ]


def _trace_rows(
    simulation: SimulationResult,
    *,
    panel: str,
    delay_ms: float,
    start_ms: float,
    stop_ms: float,
    parameters: dict[str, float],
) -> list[dict[str, object]]:
    mask = (simulation.time_ms >= start_ms) & (simulation.time_ms <= stop_ms)
    arrays = {
        "input_signal": simulation.input_signal[0],
        "dendritic_potential": simulation.dendritic_potential[0],
        "output_signal": simulation.output_signal,
        "dendritic_error": simulation.dendritic_error[0],
        "weight": simulation.weights[0],
    }
    rows: list[dict[str, object]] = []
    for variable, values in arrays.items():
        for time_ms, value in zip(simulation.time_ms[mask], values[mask]):
            rows.append(
                {
                    "panel": panel,
                    "series": f"delay = {delay_ms:g} ms",
                    "variable": variable,
                    "time_ms": float(time_ms),
                    "delay_ms": float(delay_ms),
                    "value": float(value),
                    "lambda_per_ms": parameters["decay_rate"],
                    "tau_ms": parameters["transmission_delay_ms"],
                    "gamma": parameters["plasticity_threshold"],
                    "alpha": parameters["learning_rate"],
                    "initial_weight": parameters["initial_weight"],
                }
            )
    return rows


def generate_results(
    *,
    delays_ms: Sequence[float] | np.ndarray = tuple(range(-50, 50)),
    trace_delays_ms: Sequence[float] = (-20.0, 10.0),
    trace_start_ms: float = 50.0,
    trace_stop_ms: float = 150.0,
    dt_ms: float = 0.1,
    decay_rate: float = 0.1,
    transmission_delay_ms: float = 5.0,
    plasticity_threshold: float = 0.02,
    learning_rate: float = 0.01,
    initial_weight: float = 0.5,
    gate_gain: float = 200.0,
    membrane_time_constants_ms: Sequence[float] = (5.0, 10.0, 15.0),
    alpha_values: Sequence[float] = (0.02, 0.01, 0.005),
    tau_values_ms: Sequence[float] = (0.0, 3.0, 6.0),
    gamma_values: Sequence[float] = (-0.5, 0.02, 0.1, 0.5),
) -> list[dict[str, object]]:
    """Return long-form numerical results for panels 6a--h without file I/O."""

    if len(trace_delays_ms) != 2:
        raise ValueError("trace_delays_ms must contain the delays for panels 6a and 6c")
    delays = np.asarray(delays_ms, dtype=float)
    if delays.ndim != 1:
        raise ValueError("delays_ms must be one-dimensional")

    base = _parameters(
        decay_rate=decay_rate,
        transmission_delay_ms=transmission_delay_ms,
        plasticity_threshold=plasticity_threshold,
        learning_rate=learning_rate,
        initial_weight=initial_weight,
    )
    rows: list[dict[str, object]] = []

    for panel, delay_ms in zip(("6a", "6c"), trace_delays_ms):
        _, simulation = stdp_single(
            delay_ms,
            transmission_delay_ms=transmission_delay_ms,
            decay_rate=decay_rate,
            plasticity_threshold=plasticity_threshold,
            learning_rate=learning_rate,
            initial_weight=initial_weight,
            dt_ms=dt_ms,
            gate_gain=gate_gain,
            return_trace=True,
        )
        rows += _trace_rows(
            simulation,
            panel=panel,
            delay_ms=float(delay_ms),
            start_ms=trace_start_ms,
            stop_ms=trace_stop_ms,
            parameters=base,
        )

    rows += _curve_rows(
        panel="6b",
        series="Default",
        delays_ms=delays,
        parameters=base,
        dt_ms=dt_ms,
        gate_gain=gate_gain,
    )

    for time_constant_ms in membrane_time_constants_ms:
        parameters = {**base, "decay_rate": 1.0 / float(time_constant_ms)}
        rows += _curve_rows(
            panel="6d",
            series=f"1/lambda = {time_constant_ms:g} ms",
            delays_ms=delays,
            parameters=parameters,
            dt_ms=dt_ms,
            gate_gain=gate_gain,
        )
    for alpha in alpha_values:
        parameters = {**base, "learning_rate": float(alpha)}
        rows += _curve_rows(
            panel="6e",
            series=f"alpha = {alpha:g}",
            delays_ms=delays,
            parameters=parameters,
            dt_ms=dt_ms,
            gate_gain=gate_gain,
        )
    for tau_ms in tau_values_ms:
        parameters = {**base, "transmission_delay_ms": float(tau_ms)}
        rows += _curve_rows(
            panel="6f",
            series=f"tau = {tau_ms:g} ms",
            delays_ms=delays,
            parameters=parameters,
            dt_ms=dt_ms,
            gate_gain=gate_gain,
        )
    for gamma in gamma_values:
        parameters = {**base, "plasticity_threshold": float(gamma)}
        rows += _curve_rows(
            panel="6g",
            series=f"gamma = {gamma:g}",
            delays_ms=delays,
            parameters=parameters,
            dt_ms=dt_ms,
            gate_gain=gate_gain,
        )

    rows += _curve_rows(
        panel="6h",
        series="Simulation",
        delays_ms=delays,
        parameters=base,
        dt_ms=dt_ms,
        gate_gain=gate_gain,
    )
    rows += _analytic_rows(delays_ms=delays, parameters=base)
    return rows


def write_results(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_results(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    numeric = set(CSV_FIELDS).difference({"panel", "series", "variable"})
    with path.open(newline="", encoding="utf-8") as handle:
        for source_row in csv.DictReader(handle):
            row: dict[str, object] = {}
            for key, value in source_row.items():
                row[key] = float(value) if key in numeric and value != "" else value
            rows.append(row)
    return rows


def _series(rows: Sequence[dict[str, object]], panel: str) -> list[str]:
    return list(
        dict.fromkeys(str(row["series"]) for row in rows if row["panel"] == panel)
    )


def _xy(
    rows: Sequence[dict[str, object]],
    *,
    panel: str,
    series: str,
    variable: str,
    x_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["panel"] == panel
        and row["series"] == series
        and row["variable"] == variable
    ]
    selected.sort(key=lambda row: float(row[x_key]))
    return (
        np.asarray([row[x_key] for row in selected], dtype=float),
        np.asarray([row["value"] for row in selected], dtype=float),
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="white", linewidth=0.8)
    ax.set_facecolor("#eef0f5")


def _draw_trace_panel(
    rows: Sequence[dict[str, object]], panel: str, axes: Sequence[plt.Axes]
) -> None:
    series = _series(rows, panel)[0]
    for ax, variable in zip(axes, TRACE_VARIABLES):
        x, y = _xy(
            rows,
            panel=panel,
            series=series,
            variable=variable,
            x_key="time_ms",
        )
        ax.plot(x, y, color="#4c72b0", linewidth=1.35)
        ax.set_ylabel(TRACE_LABELS[variable], rotation=0, labelpad=14)
        _style_axis(ax)
    axes[-1].set_xlabel("Time [ms]")
    axes[-1].set_ylim(
        np.min(
            _xy(rows, panel=panel, series=series, variable="weight", x_key="time_ms")[1]
        )
        - 0.001,
        np.max(
            _xy(rows, panel=panel, series=series, variable="weight", x_key="time_ms")[1]
        )
        + 0.001,
    )


def plot_trace_panel(rows: Sequence[dict[str, object]], panel: str) -> plt.Figure:
    fig, axes = plt.subplots(
        5, 1, figsize=(4.0, 6.4), sharex=True, constrained_layout=True
    )
    _draw_trace_panel(rows, panel, axes)
    delay_ms = float(next(row["delay_ms"] for row in rows if row["panel"] == panel))
    axes[0].set_title(f"Pre-post delay: {delay_ms:g} ms", fontsize=9)
    return fig


def _display_series_label(series: str) -> str:
    return (
        series.replace("1/lambda", r"$1/\lambda$")
        .replace("alpha", r"$\alpha$")
        .replace("tau", r"$\tau$")
        .replace("gamma", r"$\gamma$")
    )


def _panel_colors(panel: str, count: int) -> list[str]:
    """Retain paper colors for defaults and support arbitrary CLI series."""

    base_colors = PANEL_PALETTES[panel]
    if count <= len(base_colors):
        return list(base_colors[:count])
    if count == 1:
        return [matplotlib.colors.to_hex(plt.colormaps["viridis"](0.5))]
    return [
        matplotlib.colors.to_hex(plt.colormaps["viridis"](index / (count - 1)))
        for index in range(count)
    ]


def _draw_curve_panel(
    rows: Sequence[dict[str, object]], panel: str, ax: plt.Axes
) -> None:
    series_values = _series(rows, panel)
    for series, color in zip(series_values, _panel_colors(panel, len(series_values))):
        x, y = _xy(
            rows,
            panel=panel,
            series=series,
            variable="weight_change_percent",
            x_key="delay_ms",
        )
        linestyle = "--" if panel == "6h" and "analytic" in series.lower() else "-"
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.6,
            linestyle=linestyle,
            label=_display_series_label(series),
        )
    ax.axhline(0.0, color="0.65", linewidth=0.7, zorder=0)
    ax.set_xlabel("Interval between pre and post spikes [ms]")
    ax.set_ylabel("Weight change [%]")
    ax.set_title(PANEL_TITLES[panel], fontsize=10)
    if panel != "6b":
        ax.legend(fontsize=7, frameon=True)
    _style_axis(ax)


def plot_curve_panel(rows: Sequence[dict[str, object]], panel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.1, 3.4), constrained_layout=True)
    _draw_curve_panel(rows, panel, ax)
    return fig


def plot_core_composite(rows: Sequence[dict[str, object]]) -> plt.Figure:
    """Arrange panels 6a--h in a compact composite (experimental fits omitted)."""

    fig = plt.figure(figsize=(12.0, 8.0), constrained_layout=True)
    outer = fig.add_gridspec(3, 12, height_ratios=(1.7, 1.0, 1.0))
    for panel, slot in (("6a", outer[0, 0:4]), ("6c", outer[0, 8:12])):
        inner = slot.subgridspec(5, 1, hspace=0.08)
        axes = [fig.add_subplot(inner[index, 0]) for index in range(5)]
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        _draw_trace_panel(rows, panel, axes)
        axes[0].text(
            -0.24, 1.25, panel[-1], transform=axes[0].transAxes, fontweight="bold"
        )
    ax_b = fig.add_subplot(outer[0, 4:8])
    _draw_curve_panel(rows, "6b", ax_b)
    ax_b.text(-0.18, 1.05, "b", transform=ax_b.transAxes, fontweight="bold")

    for index, panel in enumerate(("6d", "6e", "6f", "6g")):
        ax = fig.add_subplot(outer[1, index * 3 : (index + 1) * 3])
        _draw_curve_panel(rows, panel, ax)
        ax.text(-0.2, 1.05, panel[-1], transform=ax.transAxes, fontweight="bold")

    ax_h = fig.add_subplot(outer[2, 0:4])
    _draw_curve_panel(rows, "6h", ax_h)
    ax_h.text(-0.2, 1.05, "h", transform=ax_h.transAxes, fontweight="bold")
    return fig


def _save_figure(fig: plt.Figure, stem: Path, formats: Sequence[str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        output = stem.with_suffix(f".{extension}")
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved panel to {output}")
    plt.close(fig)


def _write_metadata(path: Path, args: argparse.Namespace, delays: np.ndarray) -> None:
    metadata = {
        "figure": "6a-h",
        "source_mapping": {
            "6a": "stdp_dx.ipynb, delay -20 ms",
            "6b": "stdp_dw.ipynb, default curve",
            "6c": "stdp_dx.ipynb, delay 10 ms",
            "6d-g": "stdp_dw.ipynb parameter curves",
            "6h": "rafal_analytic Eq. 37 comparison",
        },
        "parameters": {
            "dt_ms": args.dt_ms,
            "lambda_per_ms": args.decay_rate,
            "tau_ms": args.transmission_delay_ms,
            "gamma": args.gamma,
            "alpha": args.learning_rate,
            "initial_weight": args.initial_weight,
            "gate_gain": args.gate_gain,
            "delays_ms": delays.tolist(),
            "trace_delays_ms": args.trace_delays_ms,
            "trace_window_ms": [args.trace_start_ms, args.trace_stop_ms],
            "panel_d_membrane_time_constants_ms": args.membrane_time_constants_ms,
            "panel_e_alpha_values": args.alpha_values,
            "panel_f_tau_values_ms": args.tau_values_ms,
            "panel_g_gamma_values": args.gamma_values,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _delay_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0 or stop < start:
        raise ValueError("delay grid requires step > 0 and stop >= start")
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce manuscript Figure 6a-h.")
    parser.add_argument("--action", choices=("all", "generate", "plot"), default="all")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument(
        "--formats", nargs="+", choices=("pdf", "svg"), default=["pdf", "svg"]
    )
    parser.add_argument("--delay-start-ms", type=float, default=-50.0)
    parser.add_argument("--delay-stop-ms", type=float, default=49.0)
    parser.add_argument("--delay-step-ms", type=float, default=1.0)
    parser.add_argument("--trace-delays-ms", type=float, nargs=2, default=[-20.0, 10.0])
    parser.add_argument("--trace-start-ms", type=float, default=50.0)
    parser.add_argument("--trace-stop-ms", type=float, default=150.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--decay-rate", type=float, default=0.1, help="lambda in ms^-1")
    parser.add_argument(
        "--transmission-delay-ms", type=float, default=5.0, help="tau in ms"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.02, help="plasticity threshold"
    )
    parser.add_argument("--learning-rate", type=float, default=0.01, help="alpha")
    parser.add_argument("--initial-weight", type=float, default=0.5)
    parser.add_argument("--gate-gain", type=float, default=200.0)
    parser.add_argument(
        "--membrane-time-constants-ms", type=float, nargs="+", default=[5.0, 10.0, 15.0]
    )
    parser.add_argument(
        "--alpha-values", type=float, nargs="+", default=[0.02, 0.01, 0.005]
    )
    parser.add_argument(
        "--tau-values-ms", type=float, nargs="+", default=[0.0, 3.0, 6.0]
    )
    parser.add_argument(
        "--gamma-values", type=float, nargs="+", default=[-0.5, 0.02, 0.1, 0.5]
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    delays = _delay_grid(args.delay_start_ms, args.delay_stop_ms, args.delay_step_ms)
    if args.action in {"all", "generate"}:
        rows = generate_results(
            delays_ms=delays,
            trace_delays_ms=args.trace_delays_ms,
            trace_start_ms=args.trace_start_ms,
            trace_stop_ms=args.trace_stop_ms,
            dt_ms=args.dt_ms,
            decay_rate=args.decay_rate,
            transmission_delay_ms=args.transmission_delay_ms,
            plasticity_threshold=args.gamma,
            learning_rate=args.learning_rate,
            initial_weight=args.initial_weight,
            gate_gain=args.gate_gain,
            membrane_time_constants_ms=args.membrane_time_constants_ms,
            alpha_values=args.alpha_values,
            tau_values_ms=args.tau_values_ms,
            gamma_values=args.gamma_values,
        )
        write_results(rows, args.results)
        _write_metadata(args.metadata, args, delays)
        print(f"Saved numerical results to {args.results}")

    if args.action in {"all", "plot"}:
        rows = read_results(args.results)
        plt.rcParams.update(
            {
                "font.size": 9,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "pdf.fonttype": 42,
                "svg.fonttype": "none",
            }
        )
        for panel in ("6a", "6c"):
            _save_figure(
                plot_trace_panel(rows, panel),
                args.output_dir / f"figure{panel}",
                args.formats,
            )
        for panel in ("6b", "6d", "6e", "6f", "6g", "6h"):
            _save_figure(
                plot_curve_panel(rows, panel),
                args.output_dir / f"figure{panel}",
                args.formats,
            )
        _save_figure(
            plot_core_composite(rows), args.output_dir / "figure6_core", args.formats
        )


if __name__ == "__main__":
    main()
