"""Analytical approximation to the PDM STDP curve (manuscript Eq. 37).

This module is both a small reusable library and a standalone reproduction
entry point for the comparison in Fig. 6h. Data generation and plotting are
separate operations so committed CSV results can be plotted without rerunning
the numerical simulations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .model import stdp_curve
except ImportError:  # Support ``python analytic.py`` from any directory.
    from model import stdp_curve


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results" / "figure6h.csv"
DEFAULT_METADATA = HERE / "results" / "figure6h.json"
DEFAULT_PLOT_DIR = HERE / "plots"


def analytic_stdp_curve(
    delays_ms: Sequence[float] | np.ndarray,
    *,
    transmission_delay_ms: float = 5.0,
    decay_rate: float = 0.1,
    plasticity_threshold: float = 0.02,
    learning_rate: float = 0.01,
    initial_weight: float = 0.5,
    percent_change: bool = True,
) -> np.ndarray:
    """Evaluate the small-learning-rate, step-gate STDP approximation.

    ``plasticity_threshold`` is gamma in the manuscript. Equation 37 assumes
    ``0 < gamma < 1``; values outside that interval therefore raise a clear
    error instead of silently producing an invalid logarithm.
    """

    delays = np.asarray(delays_ms, dtype=float)
    if delays.ndim != 1:
        raise ValueError("delays_ms must be one-dimensional")
    if decay_rate <= 0:
        raise ValueError("decay_rate must be positive")
    if not 0.0 < plasticity_threshold < 1.0:
        raise ValueError(
            "the analytic approximation requires 0 < plasticity_threshold < 1"
        )
    if initial_weight <= 0:
        raise ValueError("initial_weight must be positive")

    start_ms = np.maximum(0.0, transmission_delay_ms - delays)
    end_ms = -np.log(plasticity_threshold) / decay_rate

    first_integral = np.exp(
        -decay_rate * (2.0 * start_ms - transmission_delay_ms + delays)
    ) - np.exp(-decay_rate * (2.0 * end_ms - transmission_delay_ms + delays))
    second_integral = np.exp(
        -2.0 * decay_rate * (start_ms - transmission_delay_ms + delays)
    ) - np.exp(-2.0 * decay_rate * (end_ms - transmission_delay_ms + delays))
    changes = (
        learning_rate
        / (2.0 * decay_rate)
        * (first_integral - initial_weight * second_integral)
    )
    changes = np.where(start_ms < end_ms, changes, 0.0)
    if percent_change:
        changes = 100.0 * changes / initial_weight
    return changes


def generate_comparison(
    delays_ms: Sequence[float] | np.ndarray,
    *,
    transmission_delay_ms: float = 5.0,
    decay_rate: float = 0.1,
    plasticity_threshold: float = 0.02,
    learning_rate: float = 0.01,
    initial_weight: float = 0.5,
    dt_ms: float = 0.1,
    gate_gain: float = 200.0,
) -> dict[str, np.ndarray]:
    """Generate simulated and analytical percentage changes for Fig. 6h."""

    delays = np.asarray(delays_ms, dtype=float)
    simulation = stdp_curve(
        delays,
        transmission_delay_ms=transmission_delay_ms,
        decay_rate=decay_rate,
        plasticity_threshold=plasticity_threshold,
        learning_rate=learning_rate,
        initial_weight=initial_weight,
        dt_ms=dt_ms,
        gate_gain=gate_gain,
        percent_change=True,
    )
    approximation = analytic_stdp_curve(
        delays,
        transmission_delay_ms=transmission_delay_ms,
        decay_rate=decay_rate,
        plasticity_threshold=plasticity_threshold,
        learning_rate=learning_rate,
        initial_weight=initial_weight,
        percent_change=True,
    )
    return {
        "delay_ms": delays,
        "simulation_percent": simulation,
        "analytic_percent": approximation,
    }


def write_comparison_csv(data: dict[str, np.ndarray], path: Path) -> None:
    """Write Fig. 6h comparison arrays to a compact, portable CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["delay_ms", "simulation_percent", "analytic_percent"])
        writer.writerows(
            zip(
                data["delay_ms"],
                data["simulation_percent"],
                data["analytic_percent"],
            )
        )


def read_comparison_csv(path: Path) -> dict[str, np.ndarray]:
    """Read a CSV created by :func:`write_comparison_csv`."""

    columns: dict[str, list[float]] = {
        "delay_ms": [],
        "simulation_percent": [],
        "analytic_percent": [],
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for name in columns:
                columns[name].append(float(row[name]))
    return {name: np.asarray(values) for name, values in columns.items()}


def plot_comparison(
    data: dict[str, np.ndarray],
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot Fig. 6h on ``ax`` and return that axes object."""

    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.4), constrained_layout=True)
    ax.plot(
        data["delay_ms"],
        data["simulation_percent"],
        color="#4c72b0",
        linewidth=1.8,
        label="Simulation",
    )
    ax.plot(
        data["delay_ms"],
        data["analytic_percent"],
        color="#c44e52",
        linewidth=1.5,
        linestyle="--",
        label="Approximate\nanalytic",
    )
    ax.axhline(0.0, color="0.65", linewidth=0.7, zorder=0)
    ax.set_xlabel("Interval between pre and post spikes [ms]")
    ax.set_ylabel("Weight change [%]")
    ax.legend(frameon=True, fontsize=8)
    return ax


def _write_metadata(path: Path, args: argparse.Namespace) -> None:
    metadata = {
        "figure": "6h",
        "description": "simulated STDP and Eq. 37 analytical approximation",
        "parameters": {
            "dt_ms": args.dt_ms,
            "lambda_per_ms": args.decay_rate,
            "tau_ms": args.transmission_delay_ms,
            "gamma": args.gamma,
            "alpha": args.learning_rate,
            "initial_weight": args.initial_weight,
            "gate_gain": args.gate_gain,
            "delay_start_ms": args.delay_start_ms,
            "delay_stop_ms": args.delay_stop_ms,
            "delay_step_ms": args.delay_step_ms,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _delay_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0 or stop < start:
        raise ValueError("delay grid requires step > 0 and stop >= start")
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the analytical comparison in manuscript Fig. 6h."
    )
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.action in {"all", "generate"}:
        delays = _delay_grid(
            args.delay_start_ms, args.delay_stop_ms, args.delay_step_ms
        )
        data = generate_comparison(
            delays,
            transmission_delay_ms=args.transmission_delay_ms,
            decay_rate=args.decay_rate,
            plasticity_threshold=args.gamma,
            learning_rate=args.learning_rate,
            initial_weight=args.initial_weight,
            dt_ms=args.dt_ms,
            gate_gain=args.gate_gain,
        )
        write_comparison_csv(data, args.results)
        _write_metadata(args.metadata, args)
        print(f"Saved numerical results to {args.results}")

    if args.action in {"all", "plot"}:
        data = read_comparison_csv(args.results)
        plt.rcParams.update(
            {
                "font.size": 9,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "pdf.fonttype": 42,
                "svg.fonttype": "none",
            }
        )
        fig, ax = plt.subplots(figsize=(4.2, 3.4), constrained_layout=True)
        plot_comparison(data, ax=ax)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for extension in args.formats:
            output = args.output_dir / f"figure6h.{extension}"
            fig.savefig(output, bbox_inches="tight")
            print(f"Saved panel to {output}")
        plt.close(fig)


if __name__ == "__main__":
    main()
