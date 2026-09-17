"""Generate manuscript Figure 7a: STDP versus initial synaptic weight."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
if __package__:
    from .model import stdp_curve
else:
    sys.path.insert(0, str(HERE))
    from model import stdp_curve

DEFAULT_RESULTS = HERE / "results" / "figure7a.csv"
DEFAULT_METADATA = HERE / "results" / "figure7a.json"
DEFAULT_PLOT_DIR = HERE / "plots"
CSV_FIELDS = (
    "panel",
    "initial_weight",
    "delay_ms",
    "weight_change_percent",
    "lambda_per_ms",
    "tau_ms",
    "gamma",
    "alpha",
)


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
    """Return Figure 7a as serializable long-form records."""

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
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                key: value if key == "panel" else float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def plot_panel(rows: Sequence[dict[str, float | str]]) -> plt.Figure:
    """Plot Figure 7a from long-form result rows."""

    weights = list(dict.fromkeys(float(row["initial_weight"]) for row in rows))
    colors = plt.cm.Greys(np.linspace(0.3, 0.85, len(weights)))
    figure, axis = plt.subplots(figsize=(4.4, 3.7), constrained_layout=True)
    for initial_weight, color in zip(weights, colors):
        selected = [
            row for row in rows if float(row["initial_weight"]) == initial_weight
        ]
        selected.sort(key=lambda row: float(row["delay_ms"]))
        axis.plot(
            [float(row["delay_ms"]) for row in selected],
            [float(row["weight_change_percent"]) for row in selected],
            color=color,
            linewidth=1.7,
            label=f"{initial_weight:g}",
        )
    axis.axhline(0.0, color="0.65", linewidth=0.7, zorder=0)
    axis.set_xlabel("Interval between pre and post spikes [ms]")
    axis.set_ylabel("Weight change [%]")
    axis.legend(title="Initial weight", frameon=True, fontsize=8, title_fontsize=8)
    axis.grid(True, color="white", linewidth=0.8)
    axis.set_facecolor("#eef0f5")
    return figure


def _delay_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0 or stop < start:
        raise ValueError("delay grid requires step > 0 and stop >= start")
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--decay-rate", type=float, default=0.1)
    parser.add_argument("--transmission-delay-ms", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--gate-gain", type=float, default=200.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        metadata = {
            "figure": "7a",
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
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved numerical results to {args.results}")

    if args.action in {"all", "plot"}:
        figure = plot_panel(read_results(args.results))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for extension in args.formats:
            output = args.output_dir / f"figure7a.{extension}"
            figure.savefig(output, bbox_inches="tight")
            print(f"Saved panel to {output}")
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
