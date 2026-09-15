"""Reproduce manuscript Figure 7a: STDP versus initial synaptic weight."""

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
except ImportError:  # Support ``python figure7.py`` from any directory.
    from model import stdp_curve


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results" / "figure7a.csv"
DEFAULT_METADATA = HERE / "results" / "figure7a.json"
DEFAULT_PLOT_DIR = HERE / "plots"
CSV_FIELDS = [
    "panel",
    "initial_weight",
    "delay_ms",
    "weight_change_percent",
    "lambda_per_ms",
    "tau_ms",
    "gamma",
    "alpha",
]


def generate_results(
    *,
    delays_ms: Sequence[float] | np.ndarray = tuple(range(-50, 50)),
    initial_weights: Sequence[float] = (0.25, 0.5, 0.75),
    dt_ms: float = 0.1,
    decay_rate: float = 0.1,
    transmission_delay_ms: float = 5.0,
    plasticity_threshold: float = 0.02,
    learning_rate: float = 0.01,
    gate_gain: float = 200.0,
) -> list[dict[str, float | str]]:
    """Return Figure 7a data as serializable long-form records."""

    delays = np.asarray(delays_ms, dtype=float)
    if delays.ndim != 1:
        raise ValueError("delays_ms must be one-dimensional")
    rows: list[dict[str, float | str]] = []
    for initial_weight in initial_weights:
        changes = stdp_curve(
            delays,
            transmission_delay_ms=transmission_delay_ms,
            decay_rate=decay_rate,
            plasticity_threshold=plasticity_threshold,
            learning_rate=learning_rate,
            initial_weight=float(initial_weight),
            dt_ms=dt_ms,
            gate_gain=gate_gain,
            percent_change=True,
        )
        for delay_ms, change in zip(delays, changes):
            rows.append(
                {
                    "panel": "7a",
                    "initial_weight": float(initial_weight),
                    "delay_ms": float(delay_ms),
                    "weight_change_percent": float(change),
                    "lambda_per_ms": float(decay_rate),
                    "tau_ms": float(transmission_delay_ms),
                    "gamma": float(plasticity_threshold),
                    "alpha": float(learning_rate),
                }
            )
    return rows


def write_results(rows: Sequence[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_results(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    key: value if key == "panel" else float(value)
                    for key, value in row.items()
                }
            )
    return rows


def plot_panel(rows: Sequence[dict[str, float | str]]) -> plt.Figure:
    """Plot Figure 7a from rows returned by :func:`read_results`."""

    weights = list(dict.fromkeys(float(row["initial_weight"]) for row in rows))
    colors = plt.cm.Greys(np.linspace(0.3, 0.85, len(weights)))
    fig, ax = plt.subplots(figsize=(4.4, 3.7), constrained_layout=True)
    for initial_weight, color in zip(weights, colors):
        selected = [
            row for row in rows if float(row["initial_weight"]) == initial_weight
        ]
        selected.sort(key=lambda row: float(row["delay_ms"]))
        ax.plot(
            [float(row["delay_ms"]) for row in selected],
            [float(row["weight_change_percent"]) for row in selected],
            color=color,
            linewidth=1.7,
            label=f"{initial_weight:g}",
        )
    ax.axhline(0.0, color="0.65", linewidth=0.7, zorder=0)
    ax.set_xlabel("Interval between pre and post spikes [ms]")
    ax.set_ylabel("Weight change [%]")
    ax.legend(title="Initial weight", frameon=True, fontsize=8, title_fontsize=8)
    ax.grid(True, color="white", linewidth=0.8)
    ax.set_facecolor("#eef0f5")
    return fig


def _write_metadata(path: Path, args: argparse.Namespace, delays: np.ndarray) -> None:
    metadata = {
        "figure": "7a",
        "description": "effect of initial weight on the simulated STDP curve",
        "parameters": {
            "dt_ms": args.dt_ms,
            "lambda_per_ms": args.decay_rate,
            "tau_ms": args.transmission_delay_ms,
            "gamma": args.gamma,
            "alpha": args.learning_rate,
            "initial_weights": args.initial_weights,
            "delays_ms": delays.tolist(),
            "gate_gain": args.gate_gain,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _delay_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0 or stop < start:
        raise ValueError("delay grid requires step > 0 and stop >= start")
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce manuscript Figure 7a.")
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
    parser.add_argument(
        "--initial-weights", type=float, nargs="+", default=[0.25, 0.5, 0.75]
    )
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--decay-rate", type=float, default=0.1, help="lambda in ms^-1")
    parser.add_argument(
        "--transmission-delay-ms", type=float, default=5.0, help="tau in ms"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.02, help="plasticity threshold"
    )
    parser.add_argument("--learning-rate", type=float, default=0.01, help="alpha")
    parser.add_argument("--gate-gain", type=float, default=200.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    delays = _delay_grid(args.delay_start_ms, args.delay_stop_ms, args.delay_step_ms)
    if args.action in {"all", "generate"}:
        rows = generate_results(
            delays_ms=delays,
            initial_weights=args.initial_weights,
            dt_ms=args.dt_ms,
            decay_rate=args.decay_rate,
            transmission_delay_ms=args.transmission_delay_ms,
            plasticity_threshold=args.gamma,
            learning_rate=args.learning_rate,
            gate_gain=args.gate_gain,
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
        fig = plot_panel(rows)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for extension in args.formats:
            output = args.output_dir / f"figure7a.{extension}"
            fig.savefig(output, bbox_inches="tight")
            print(f"Saved panel to {output}")
        plt.close(fig)


if __name__ == "__main__":
    main()
