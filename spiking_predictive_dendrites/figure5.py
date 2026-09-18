"""Reproduce the numerical panels of manuscript Figure 5.

The script writes a long-form CSV before plotting. Use ``--action generate``
for numerical generation and ``--action plot`` to render an existing CSV.
The historical source used alpha=0.2 for the displayed trace, whereas the
manuscript caption reports alpha=0.1; both auditable parameter sets are
available through ``--parameter-set``.

The source plotting helper displayed ``v^d - x^out`` under the dendritic-error
label even though its update rule and the manuscript define
``epsilon^d = x^out - v^d``. This reproduction uses the manuscript sign for
the displayed error; the simulated weight dynamics are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    from .model import SimulationResult, simulate_two_dendrites
except ImportError:  # Support ``python figure5.py`` from any directory.
    from model import SimulationResult, simulate_two_dendrites


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results" / "figure5.csv"
DEFAULT_METADATA = HERE / "results" / "figure5.json"
DEFAULT_PLOT_DIR = HERE / "plots"

COLORS = {
    "forward": "#4c72b0",
    "backward": "#dd8452",
    "soma": "#55a868",
}
CSV_FIELDS = [
    "panel",
    "section",
    "series",
    "variable",
    "time_ms",
    "value",
    "lambda_per_ms",
    "tau_ms",
    "gamma",
    "alpha",
    "initial_weight",
]


def _input_spikes(dt_ms: float, duration_ms: float) -> np.ndarray:
    """Return the deterministic spike pattern used by the source notebook."""

    n_steps = int(duration_ms / dt_ms)
    spikes = np.zeros((2, n_steps), dtype=float)
    for dendrite, times_ms in enumerate(((10.0, 40.0), (20.0, 23.0, 26.0, 28.0, 30.0))):
        for time_ms in times_ms:
            index = int(round(time_ms / dt_ms))
            if index >= n_steps:
                raise ValueError(
                    "duration_ms does not include the Figure 5 spike pattern"
                )
            spikes[dendrite, index] = 1.0
    return spikes


def simulate_figure5(
    *,
    dt_ms: float = 0.1,
    duration_ms: float = 1000.0,
    decay_rate: float = 0.1,
    transmission_delay_ms: float = 5.0,
    plasticity_threshold: float = 0.02,
    learning_rate: float = 0.2,
    initial_weights: Sequence[float] = (0.6, 0.3),
    gate_gain: float = 200.0,
) -> SimulationResult:
    """Run the deterministic two-input simulation used in Figure 5b/c."""

    return simulate_two_dendrites(
        _input_spikes(dt_ms, duration_ms),
        initial_weights=initial_weights,
        dt_ms=dt_ms,
        transmission_delay_ms=transmission_delay_ms,
        decay_rate=decay_rate,
        plasticity_threshold=plasticity_threshold,
        learning_rate=learning_rate,
        gate_gain=gate_gain,
    )


def _trace_rows(
    simulation: SimulationResult,
    *,
    panel: str,
    section: str,
    series: str,
    variables: Iterable[tuple[str, np.ndarray]],
    start_ms: float,
    stop_ms: float,
    parameters: dict[str, float],
) -> list[dict[str, object]]:
    mask = (simulation.time_ms >= start_ms) & (simulation.time_ms < stop_ms)
    times = simulation.time_ms[mask]
    rows: list[dict[str, object]] = []
    for variable, values in variables:
        for time_ms, value in zip(times, np.asarray(values)[mask]):
            rows.append(
                {
                    "panel": panel,
                    "section": section,
                    "series": series,
                    "variable": variable,
                    "time_ms": float(time_ms),
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
    parameter_set: str = "historical",
    learning_rate: float | None = None,
    dt_ms: float = 0.1,
    duration_ms: float = 1000.0,
    decay_rate: float = 0.1,
    transmission_delay_ms: float = 5.0,
    plasticity_threshold: float = 0.02,
    initial_weights: Sequence[float] = (0.6, 0.3),
    gate_gain: float = 200.0,
    plot_start_ms: float = 6.0,
    plot_stop_ms: float = 50.0,
    membrane_time_constants_ms: Sequence[float] = (5.0, 10.0, 15.0),
    alpha_values: Sequence[float] = (0.02, 0.01, 0.005),
    tau_values_ms: Sequence[float] = (0.0, 5.0, 10.0),
    gamma_values: Sequence[float] = (-0.5, 0.02, 0.5),
) -> list[dict[str, object]]:
    """Generate all time series for Figure 5b/c as long-form records."""

    if parameter_set not in {"historical", "manuscript"}:
        raise ValueError("parameter_set must be 'historical' or 'manuscript'")
    if learning_rate is None:
        learning_rate = 0.2 if parameter_set == "historical" else 0.1
    if len(initial_weights) != 2:
        raise ValueError("initial_weights must contain two values")

    base_parameters = {
        "decay_rate": float(decay_rate),
        "transmission_delay_ms": float(transmission_delay_ms),
        "plasticity_threshold": float(plasticity_threshold),
        "learning_rate": float(learning_rate),
        "initial_weight": float(initial_weights[0]),
    }
    baseline = simulate_figure5(
        dt_ms=dt_ms,
        duration_ms=duration_ms,
        decay_rate=decay_rate,
        transmission_delay_ms=transmission_delay_ms,
        plasticity_threshold=plasticity_threshold,
        learning_rate=learning_rate,
        initial_weights=initial_weights,
        gate_gain=gate_gain,
    )
    rows = _trace_rows(
        baseline,
        panel="5b",
        section="baseline",
        series="forward",
        variables=(
            ("input_spikes", baseline.input_spikes[0]),
            ("input_signal", baseline.input_signal[0]),
            ("dendritic_potential", baseline.dendritic_potential[0]),
            ("dendritic_error", baseline.dendritic_error[0]),
            ("weight", baseline.weights[0]),
        ),
        start_ms=plot_start_ms,
        stop_ms=plot_stop_ms,
        parameters=base_parameters,
    )
    rows += _trace_rows(
        baseline,
        panel="5b",
        section="baseline",
        series="backward",
        variables=(
            ("input_spikes", baseline.input_spikes[1]),
            ("input_signal", baseline.input_signal[1]),
            ("dendritic_potential", baseline.dendritic_potential[1]),
            ("dendritic_error", baseline.dendritic_error[1]),
            ("weight", baseline.weights[1]),
        ),
        start_ms=plot_start_ms,
        stop_ms=plot_stop_ms,
        parameters={**base_parameters, "initial_weight": float(initial_weights[1])},
    )
    rows += _trace_rows(
        baseline,
        panel="5b",
        section="baseline",
        series="soma",
        variables=(
            ("somatic_potential", baseline.somatic_potential),
            ("output_spikes", baseline.output_spikes),
            ("output_signal", baseline.output_signal),
            ("energy", baseline.energy),
        ),
        start_ms=plot_start_ms,
        stop_ms=plot_stop_ms,
        parameters=base_parameters,
    )

    for time_constant_ms in membrane_time_constants_ms:
        scenario_lambda = 1.0 / float(time_constant_ms)
        parameters = {**base_parameters, "decay_rate": scenario_lambda}
        simulation = simulate_figure5(
            dt_ms=dt_ms,
            duration_ms=duration_ms,
            decay_rate=scenario_lambda,
            transmission_delay_ms=transmission_delay_ms,
            plasticity_threshold=plasticity_threshold,
            learning_rate=learning_rate,
            initial_weights=initial_weights,
            gate_gain=gate_gain,
        )
        rows += _trace_rows(
            simulation,
            panel="5c",
            section="lambda",
            series=f"1/lambda = {time_constant_ms:g} ms",
            variables=(("input_signal", simulation.input_signal[0]),),
            start_ms=plot_start_ms,
            stop_ms=plot_stop_ms,
            parameters=parameters,
        )

    for alpha in alpha_values:
        parameters = {**base_parameters, "learning_rate": float(alpha)}
        simulation = simulate_figure5(
            dt_ms=dt_ms,
            duration_ms=duration_ms,
            decay_rate=decay_rate,
            transmission_delay_ms=transmission_delay_ms,
            plasticity_threshold=plasticity_threshold,
            learning_rate=alpha,
            initial_weights=initial_weights,
            gate_gain=gate_gain,
        )
        rows += _trace_rows(
            simulation,
            panel="5c",
            section="alpha",
            series=f"alpha = {alpha:g}",
            variables=(("weight", simulation.weights[0]),),
            start_ms=plot_start_ms,
            stop_ms=plot_stop_ms,
            parameters=parameters,
        )

    for tau_ms in tau_values_ms:
        parameters = {**base_parameters, "transmission_delay_ms": float(tau_ms)}
        simulation = simulate_figure5(
            dt_ms=dt_ms,
            duration_ms=duration_ms,
            decay_rate=decay_rate,
            transmission_delay_ms=tau_ms,
            plasticity_threshold=plasticity_threshold,
            learning_rate=learning_rate,
            initial_weights=initial_weights,
            gate_gain=gate_gain,
        )
        rows += _trace_rows(
            simulation,
            panel="5c",
            section="tau",
            series=f"tau = {tau_ms:g} ms",
            variables=(
                ("input_spikes", simulation.input_spikes[0]),
                ("dendritic_potential", simulation.dendritic_potential[0]),
            ),
            start_ms=plot_start_ms,
            stop_ms=plot_stop_ms,
            parameters=parameters,
        )

    for gamma in gamma_values:
        parameters = {**base_parameters, "plasticity_threshold": float(gamma)}
        simulation = simulate_figure5(
            dt_ms=dt_ms,
            duration_ms=duration_ms,
            decay_rate=decay_rate,
            transmission_delay_ms=transmission_delay_ms,
            plasticity_threshold=gamma,
            learning_rate=learning_rate,
            initial_weights=initial_weights,
            gate_gain=gate_gain,
        )
        rows += _trace_rows(
            simulation,
            panel="5c",
            section="gamma",
            series=f"gamma = {gamma:g}",
            variables=(
                ("output_signal", simulation.output_signal),
                ("weight", simulation.weights[0]),
            ),
            start_ms=plot_start_ms,
            stop_ms=plot_stop_ms,
            parameters=parameters,
        )
    return rows


def write_results(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_results(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    numeric = set(CSV_FIELDS).difference({"panel", "section", "series", "variable"})
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    key: float(value) if key in numeric else value
                    for key, value in row.items()
                }
            )
    return rows


def _xy(
    rows: Sequence[dict[str, object]],
    *,
    panel: str,
    section: str,
    series: str,
    variable: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["panel"] == panel
        and row["section"] == section
        and row["series"] == series
        and row["variable"] == variable
    ]
    selected.sort(key=lambda row: float(row["time_ms"]))
    return (
        np.asarray([row["time_ms"] for row in selected], dtype=float),
        np.asarray([row["value"] for row in selected], dtype=float),
    )


def _available_series(
    rows: Sequence[dict[str, object]], panel: str, section: str
) -> list[str]:
    return list(
        dict.fromkeys(
            str(row["series"])
            for row in rows
            if row["panel"] == panel and row["section"] == section
        )
    )


def _series_colors(base_colors: Sequence[str], count: int) -> list[str]:
    """Keep manuscript colors for defaults and scale safely for custom series."""

    if count <= len(base_colors):
        return list(base_colors[:count])
    if count == 1:
        return [matplotlib.colors.to_hex(plt.colormaps["viridis"](0.5))]
    return [
        matplotlib.colors.to_hex(plt.colormaps["viridis"](index / (count - 1)))
        for index in range(count)
    ]


def plot_panel_b(rows: Sequence[dict[str, object]]) -> plt.Figure:
    """Create the nine-trace numerical panel shown in Figure 5b."""

    labels = [
        ("input_spikes", "Input spikes", r"$s^d$"),
        ("input_signal", "Input signal", r"$x^d$"),
        ("dendritic_potential", "Dendritic potential", r"$v^d$"),
        ("somatic_potential", "Somatic potential", r"$v$"),
        ("output_spikes", "Output spikes", r"$s^{out}$"),
        ("output_signal", "Output signal", r"$x^{out}$"),
        ("dendritic_error", "Dendritic errors", r"$\varepsilon^d$"),
        ("energy", "Energy", r"$E$"),
        ("weight", "Synaptic weights", r"$w^d$"),
    ]
    fig, axes = plt.subplots(
        len(labels), 1, figsize=(5.1, 9.0), sharex=True, constrained_layout=True
    )
    for index, (variable, title, symbol) in enumerate(labels):
        ax = axes[index]
        series_to_plot = (
            ("soma",)
            if variable
            in {"somatic_potential", "output_spikes", "output_signal", "energy"}
            else ("forward", "backward")
        )
        for series in series_to_plot:
            x, y = _xy(
                rows,
                panel="5b",
                section="baseline",
                series=series,
                variable=variable,
            )
            ax.plot(x, y, color=COLORS[series], linewidth=1.25, label=series)
        ax.set_ylabel(symbol, rotation=0, labelpad=15)
        ax.text(
            -0.27,
            0.5,
            title,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )
        ax.grid(True, color="white", linewidth=0.8)
        ax.set_facecolor("#eef0f5")
        if variable in {"input_spikes", "output_spikes"}:
            ax.set_ylim(-0.05, 1.05)
        if variable == "somatic_potential":
            ax.axhline(0.5, color="0.5", linestyle="--", linewidth=0.7)
    legend_handles = [
        Line2D([], [], color=COLORS[series], label=series.capitalize())
        for series in ("forward", "backward", "soma")
    ]
    axes[0].legend(
        handles=legend_handles,
        title="From",
        fontsize=7,
        title_fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    axes[-1].set_xlabel("Time [ms]")
    return fig


def plot_panel_c(rows: Sequence[dict[str, object]]) -> plt.Figure:
    """Create the four parameter-effect displays shown in Figure 5c."""

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(4.3, 8.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.2, 0.65, 1.0, 1.0, 1.0]},
        constrained_layout=True,
    )
    palettes = {
        "lambda": ["#b7ddb0", "#6fba75", "#287a45"],
        "alpha": ["#fcbba1", "#fb6a4a", "#cb181d"],
        "tau": ["#bdd7e7", "#6baed6", "#2171b5"],
        "gamma": ["#dadaeb", "#9e9ac8", "#54278f"],
    }

    lambda_series = _available_series(rows, "5c", "lambda")
    for series, color in zip(
        lambda_series, _series_colors(palettes["lambda"], len(lambda_series))
    ):
        x, y = _xy(
            rows, panel="5c", section="lambda", series=series, variable="input_signal"
        )
        axes[0].plot(
            x, y, label=series.replace("1/lambda", r"$1/\lambda$"), color=color
        )
    axes[0].set_ylabel(r"$x_i^d$", rotation=0, labelpad=12)
    axes[0].set_title(r"$\dot{x}_i^d=s_i^d-\lambda x_i^d$", loc="left", fontsize=10)
    axes[0].legend(fontsize=7, loc="upper right")

    alpha_series = _available_series(rows, "5c", "alpha")
    for series, color in zip(
        alpha_series, _series_colors(palettes["alpha"], len(alpha_series))
    ):
        x, y = _xy(rows, panel="5c", section="alpha", series=series, variable="weight")
        axes[1].plot(x, y, label=series.replace("alpha", r"$\alpha$"), color=color)
    axes[1].set_ylabel(r"$w_i^d$", rotation=0, labelpad=12)
    axes[1].legend(fontsize=7, loc="upper right")

    tau_series = _available_series(rows, "5c", "tau")
    x, y = _xy(
        rows, panel="5c", section="tau", series=tau_series[0], variable="input_spikes"
    )
    axes[2].plot(x, y, color="black", linewidth=1.0)
    axes[2].set_ylabel(r"$s_i^d$", rotation=0, labelpad=12)
    for series, color in zip(
        tau_series, _series_colors(palettes["tau"], len(tau_series))
    ):
        x, y = _xy(
            rows,
            panel="5c",
            section="tau",
            series=series,
            variable="dendritic_potential",
        )
        axes[3].plot(x, y, label=series.replace("tau", r"$\tau$"), color=color)
    axes[3].set_ylabel(r"$v^d$", rotation=0, labelpad=12)
    axes[3].legend(fontsize=7, loc="upper right")

    gamma_series = _available_series(rows, "5c", "gamma")
    x, y = _xy(
        rows,
        panel="5c",
        section="gamma",
        series=gamma_series[0],
        variable="output_signal",
    )
    axes[4].plot(x, y, color="black", linewidth=1.1)
    axes[4].set_ylabel(r"$x^{out}$", rotation=0, labelpad=12)
    for series, color in zip(
        gamma_series, _series_colors(palettes["gamma"], len(gamma_series))
    ):
        x, y = _xy(rows, panel="5c", section="gamma", series=series, variable="weight")
        axes[5].plot(x, y, label=series.replace("gamma", r"$\gamma$"), color=color)
    axes[5].set_ylabel(r"$w_i^d$", rotation=0, labelpad=12)
    axes[5].legend(fontsize=7, loc="upper right")

    for ax in axes:
        ax.grid(True, color="white", linewidth=0.8)
        ax.set_facecolor("#eef0f5")
    axes[-1].set_xlabel("Time [ms]")
    return fig


def _save_figure(fig: plt.Figure, stem: Path, formats: Sequence[str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        output = stem.with_suffix(f".{extension}")
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved panel to {output}")
    plt.close(fig)


def _write_metadata(path: Path, args: argparse.Namespace, learning_rate: float) -> None:
    metadata = {
        "figure": "5b-c",
        "visualization_note": (
            "Dendritic error is plotted as x_out - v_d, matching the "
            "manuscript and weight update. The source Figure 5 plotting helper "
            "used the opposite display sign; numerical dynamics are unchanged."
        ),
        "parameter_set": args.parameter_set,
        "parameter_set_note": (
            "historical source value alpha=0.2"
            if args.parameter_set == "historical" and args.learning_rate is None
            else "manuscript caption value alpha=0.1"
            if args.parameter_set == "manuscript" and args.learning_rate is None
            else "explicit command-line alpha override"
        ),
        "parameters": {
            "dt_ms": args.dt_ms,
            "duration_ms": args.duration_ms,
            "lambda_per_ms": args.decay_rate,
            "tau_ms": args.transmission_delay_ms,
            "gamma": args.gamma,
            "alpha": learning_rate,
            "initial_weights": args.initial_weights,
            "gate_gain": args.gate_gain,
            "plot_window_ms": [args.plot_start_ms, args.plot_stop_ms],
            "panel_c_membrane_time_constants_ms": args.membrane_time_constants_ms,
            "panel_c_alpha_values": args.alpha_values,
            "panel_c_tau_values_ms": args.tau_values_ms,
            "panel_c_gamma_values": args.gamma_values,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the numerical panels of manuscript Figure 5."
    )
    parser.add_argument("--action", choices=("all", "generate", "plot"), default="all")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument(
        "--formats", nargs="+", choices=("pdf", "svg"), default=["pdf", "svg"]
    )
    parser.add_argument(
        "--parameter-set", choices=("historical", "manuscript"), default="historical"
    )
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--duration-ms", type=float, default=1000.0)
    parser.add_argument("--decay-rate", type=float, default=0.1, help="lambda in ms^-1")
    parser.add_argument(
        "--transmission-delay-ms", type=float, default=5.0, help="tau in ms"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.02, help="plasticity threshold"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="alpha override"
    )
    parser.add_argument("--initial-weights", type=float, nargs=2, default=[0.6, 0.3])
    parser.add_argument("--gate-gain", type=float, default=200.0)
    parser.add_argument("--plot-start-ms", type=float, default=6.0)
    parser.add_argument("--plot-stop-ms", type=float, default=50.0)
    parser.add_argument(
        "--membrane-time-constants-ms", type=float, nargs="+", default=[5.0, 10.0, 15.0]
    )
    parser.add_argument(
        "--alpha-values", type=float, nargs="+", default=[0.02, 0.01, 0.005]
    )
    parser.add_argument(
        "--tau-values-ms", type=float, nargs="+", default=[0.0, 5.0, 10.0]
    )
    parser.add_argument(
        "--gamma-values", type=float, nargs="+", default=[-0.5, 0.02, 0.5]
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    effective_learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else (0.2 if args.parameter_set == "historical" else 0.1)
    )
    if args.action in {"all", "generate"}:
        rows = generate_results(
            parameter_set=args.parameter_set,
            learning_rate=args.learning_rate,
            dt_ms=args.dt_ms,
            duration_ms=args.duration_ms,
            decay_rate=args.decay_rate,
            transmission_delay_ms=args.transmission_delay_ms,
            plasticity_threshold=args.gamma,
            initial_weights=args.initial_weights,
            gate_gain=args.gate_gain,
            plot_start_ms=args.plot_start_ms,
            plot_stop_ms=args.plot_stop_ms,
            membrane_time_constants_ms=args.membrane_time_constants_ms,
            alpha_values=args.alpha_values,
            tau_values_ms=args.tau_values_ms,
            gamma_values=args.gamma_values,
        )
        write_results(rows, args.results)
        _write_metadata(args.metadata, args, effective_learning_rate)
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
        _save_figure(plot_panel_b(rows), args.output_dir / "figure5b", args.formats)
        _save_figure(plot_panel_c(rows), args.output_dir / "figure5c", args.formats)


if __name__ == "__main__":
    main()
