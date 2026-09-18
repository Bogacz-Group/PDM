#!/usr/bin/env python3
"""Generate and plot Supplementary Fig. 3d spiking weight trajectories.

Code-grounded defaults from the source experiment are one hidden unit,
``theta=0.4``, initial output-to-hidden weight 0.5, forward and backward
learning rates 0.005, and 64 passes through the four XOR patterns.  The
manuscript caption does not separately state theta or these learning rates for
panel 3d; they come from ``simulation_features.py`` and
``simulation_XOR.py``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

try:  # Support both ``python supplementary_figure3d.py`` and module imports.
    from .spiking_xor import (
        DEFAULT_DECAY,
        DEFAULT_DELAY,
        DEFAULT_DT,
        DEFAULT_LEARNING_RATE,
        DEFAULT_RESET,
        DEFAULT_THRESHOLD,
        DEFAULT_TMAX,
        initialize_weights,
        simulate,
    )
except ImportError:  # pragma: no cover - exercised by the public CLI path
    from spiking_xor import (
        DEFAULT_DECAY,
        DEFAULT_DELAY,
        DEFAULT_DT,
        DEFAULT_LEARNING_RATE,
        DEFAULT_RESET,
        DEFAULT_THRESHOLD,
        DEFAULT_TMAX,
        initialize_weights,
        simulate,
    )

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "results" / "supplementary_figure3d.csv"
PDF_PATH = SCRIPT_DIR / "plots" / "supplementary_figure3d.pdf"
SVG_PATH = SCRIPT_DIR / "plots" / "supplementary_figure3d.svg"

GRID_MIN = -1.0
GRID_MAX = 1.0
GRID_STEP = 0.25
THETA = 0.4
INITIAL_BACKWARD_WEIGHT = 0.5
NUM_DATASET_ITERATIONS = 64
SEED = 0
GATE_GAIN = 200.0
HIDDEN_GATE_EPSILON = 0.05
OUTPUT_GATE_EPSILON = -1.0

XOR_INPUTS = ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
XOR_TARGETS = (0.0, 1.0, 1.0, 0.0)

CSV_COLUMNS = (
    "panel",
    "model",
    "seed",
    "initial_wf1",
    "initial_wf2",
    "initial_wb",
    "theta",
    "learning_rate",
    "backward_learning_rate",
    "dt",
    "tmax",
    "delay",
    "output_delay_std",
    "decay_per_ms",
    "spike_threshold",
    "reset_potential",
    "hidden_gate_epsilon",
    "output_gate_epsilon",
    "gate_gain",
    "dataset_iteration",
    "source_iteration_index",
    "sample_updates",
    "wf1",
    "wf2",
    "wb",
)


def grid_values(
    minimum: float = GRID_MIN,
    maximum: float = GRID_MAX,
    step: float = GRID_STEP,
) -> np.ndarray:
    """Return an inclusive, floating-point-safe initial-weight grid."""

    minimum = float(minimum)
    maximum = float(maximum)
    step = float(step)
    if step <= 0.0:
        raise ValueError("grid step must be positive")
    if maximum < minimum:
        raise ValueError("grid maximum must be greater than or equal to minimum")
    count = int(np.floor((maximum - minimum) / step + 1e-12)) + 1
    values = minimum + step * np.arange(count, dtype=np.float64)
    if values[-1] < maximum - 1e-10:
        values = np.append(values, maximum)
    values[np.isclose(values, 0.0, atol=1e-14)] = 0.0
    return values


def run_trajectory(
    initial_wf1: float,
    initial_wf2: float,
    *,
    initial_backward_weight: float = INITIAL_BACKWARD_WEIGHT,
    theta: float = THETA,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    backward_learning_rate: float = DEFAULT_LEARNING_RATE,
    num_dataset_iterations: int = NUM_DATASET_ITERATIONS,
    seed: int = SEED,
    dt: float = DEFAULT_DT,
    tmax: float = DEFAULT_TMAX,
    delay: float = DEFAULT_DELAY,
    output_delay_std: float = 0.0,
) -> dict[str, object]:
    """Train one spiking initial condition and return scalar/list fields."""

    num_dataset_iterations = int(num_dataset_iterations)
    if num_dataset_iterations < 0:
        raise ValueError("num_dataset_iterations must be non-negative")
    if not 0.0 <= float(theta) <= 1.0:
        raise ValueError("theta must be in [0, 1]")
    if float(learning_rate) < 0.0 or float(backward_learning_rate) < 0.0:
        raise ValueError("learning rates must be non-negative")
    if float(output_delay_std) < 0.0:
        raise ValueError("output_delay_std must be non-negative")

    seed = int(seed)
    rng = np.random.default_rng(seed)
    forward_input, forward_output, backward = initialize_weights(1, rng=rng)
    forward_input[0, 0] = float(initial_wf1)
    forward_input[0, 1] = float(initial_wf2)
    backward[0, 0] = float(initial_backward_weight)

    iterations: list[int] = []
    wf1s: list[float] = []
    wf2s: list[float] = []
    wbs: list[float] = []
    last_output = 0.0
    last_dendritic_error = 0.0

    for dataset_iteration in range(1, num_dataset_iterations + 1):
        for inputs, target in zip(XOR_INPUTS, XOR_TARGETS):
            (
                forward_input,
                forward_output,
                backward,
                last_output,
                last_dendritic_error,
            ) = simulate(
                inputs,
                target,
                forward_input,
                forward_output,
                backward,
                True,
                float(theta),
                dt=float(dt),
                learning_rate=float(learning_rate),
                backward_learning_rate=float(backward_learning_rate),
                output_delay_std=float(output_delay_std),
                tmax=float(tmax),
                delay=float(delay),
                rng=rng,
            )
        iterations.append(dataset_iteration)
        wf1s.append(float(forward_input[0, 0]))
        wf2s.append(float(forward_input[0, 1]))
        wbs.append(float(backward[0, 0]))

    if wf1s:
        final_wf1, final_wf2, final_wb = wf1s[-1], wf2s[-1], wbs[-1]
    else:
        final_wf1 = float(initial_wf1)
        final_wf2 = float(initial_wf2)
        final_wb = float(initial_backward_weight)
    displacement = float(
        np.hypot(final_wf1 - float(initial_wf1), final_wf2 - float(initial_wf2))
    )

    return {
        "panel": "supplementary_figure3d",
        "model": "spiking",
        "seed": seed,
        "initial_wf1": float(initial_wf1),
        "initial_wf2": float(initial_wf2),
        "initial_wb": float(initial_backward_weight),
        "theta": float(theta),
        "learning_rate": float(learning_rate),
        "backward_learning_rate": float(backward_learning_rate),
        "dt": float(dt),
        "tmax": float(tmax),
        "delay": float(delay),
        "output_delay_std": float(output_delay_std),
        "decay_per_ms": DEFAULT_DECAY,
        "spike_threshold": DEFAULT_THRESHOLD,
        "reset_potential": DEFAULT_RESET,
        "hidden_gate_epsilon": HIDDEN_GATE_EPSILON,
        "output_gate_epsilon": OUTPUT_GATE_EPSILON,
        "gate_gain": GATE_GAIN,
        "num_dataset_iterations": num_dataset_iterations,
        "sample_updates": num_dataset_iterations * len(XOR_TARGETS),
        "iterations": iterations,
        "wf1s": wf1s,
        "wf2s": wf2s,
        "wbs": wbs,
        "final_wf1": final_wf1,
        "final_wf2": final_wf2,
        "final_wb": final_wb,
        "weight_displacement": displacement,
        "last_output": float(last_output),
        "last_dendritic_error": float(last_dendritic_error),
    }


def spiking_trajectory_trainable(
    config: Mapping[str, object],
) -> dict[str, object]:
    """mini-radas-compatible pure trainable for one initial condition."""

    return run_trajectory(
        initial_wf1=float(config.get("initial_wf1", config.get("wf11", 0.1))),
        initial_wf2=float(config.get("initial_wf2", config.get("wf12", 0.1))),
        initial_backward_weight=float(
            config.get("initial_backward_weight", config.get("wb", 0.5))
        ),
        theta=float(config.get("theta", THETA)),
        learning_rate=float(config.get("learning_rate", config.get("lr", 0.005))),
        backward_learning_rate=float(
            config.get("backward_learning_rate", config.get("lr_w2b", 0.005))
        ),
        num_dataset_iterations=int(
            config.get("num_dataset_iterations", config.get("ITER", 64))
        ),
        seed=int(config.get("seed", SEED)),
        dt=float(config.get("dt", DEFAULT_DT)),
        tmax=float(config.get("tmax", DEFAULT_TMAX)),
        delay=float(config.get("delay", DEFAULT_DELAY)),
        output_delay_std=float(
            config.get(
                "output_delay_std",
                config.get("spikes_output_delay_std", 0.0),
            )
        ),
    )


def trainable(config: Mapping[str, object]) -> dict[str, object]:
    """Backward-compatible short alias for ``spiking_trajectory_trainable``."""

    return spiking_trajectory_trainable(config)


def trajectory_rows(result: Mapping[str, object]) -> Iterator[dict[str, object]]:
    """Explode one trainable result into paper-ready, long-form rows."""

    iterations = list(result["iterations"])
    wf1s = list(result["wf1s"])
    wf2s = list(result["wf2s"])
    wbs = list(result["wbs"])
    if not (len(iterations) == len(wf1s) == len(wf2s) == len(wbs)):
        raise ValueError("trajectory fields have inconsistent lengths")
    scalar_keys = CSV_COLUMNS[: CSV_COLUMNS.index("dataset_iteration")]
    for dataset_iteration, wf1, wf2, wb in zip(iterations, wf1s, wf2s, wbs):
        row = {key: result[key] for key in scalar_keys}
        row.update(
            {
                "dataset_iteration": int(dataset_iteration),
                "source_iteration_index": int(dataset_iteration) - 1,
                "sample_updates": int(dataset_iteration) * len(XOR_TARGETS),
                "wf1": float(wf1),
                "wf2": float(wf2),
                "wb": float(wb),
            }
        )
        yield row


def run_grid(
    initial_pairs: Sequence[tuple[float, float]],
    **trajectory_kwargs: object,
) -> Iterator[dict[str, object]]:
    """Run all initial conditions sequentially and yield long-form rows."""

    for pair_index, (initial_wf1, initial_wf2) in enumerate(initial_pairs, start=1):
        print(
            f"Trajectory {pair_index}/{len(initial_pairs)}: "
            f"wf1={initial_wf1:g}, wf2={initial_wf2:g}"
        )
        result = run_trajectory(
            float(initial_wf1),
            float(initial_wf2),
            **trajectory_kwargs,
        )
        yield from trajectory_rows(result)


def write_grid_csv(
    output_path: Path,
    initial_pairs: Sequence[tuple[float, float]],
    **trajectory_kwargs: object,
) -> int:
    """Run the sequential public reproduction and write its CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in run_grid(initial_pairs, **trajectory_kwargs):
            writer.writerow(row)
            row_count += 1
    return row_count


def add_feature_regions(ax, *, threshold: float = DEFAULT_THRESHOLD) -> None:
    """Overlay exclusive-spiking regions derived in Supplementary Fig. 3b."""

    threshold = float(threshold)
    # Active only for (0, 1): wf2 >= threshold and wf1 + wf2 < threshold.
    blue = [
        (-1.0, threshold),
        (-1.0, 1.0),
        (threshold - 1.0, 1.0),
        (0.0, threshold),
        (-1.0, threshold),
    ]
    # Active only for (1, 0): wf1 >= threshold and wf1 + wf2 < threshold.
    green = [
        (threshold, 0.0),
        (1.0, threshold - 1.0),
        (1.0, -1.0),
        (threshold, -1.0),
        (threshold, 0.0),
    ]
    for vertices, color in ((blue, "#00a6ca"), (green, "#009e73")):
        xs, ys = zip(*vertices)
        ax.plot(xs, ys, color=color, linestyle="--", linewidth=2.0, zorder=4)


def plot_results(
    results_path: Path = RESULTS_PATH,
    *,
    pdf_path: Path = PDF_PATH,
    svg_path: Path = SVG_PATH,
    show_feature_regions: bool = True,
) -> None:
    """Plot a generated CSV and save both vector output formats."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import pandas as pd

    frame = pd.read_csv(results_path)
    required = {
        "seed",
        "initial_wf1",
        "initial_wf2",
        "dataset_iteration",
        "wf1",
        "wf2",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {results_path}: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"No rows in {results_path}")

    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    norm = Normalize(
        vmin=float(frame["dataset_iteration"].min()),
        vmax=float(frame["dataset_iteration"].max()),
    )
    groups = frame.groupby(["seed", "initial_wf1", "initial_wf2"], sort=False)
    for _, trajectory in groups:
        ax.scatter(
            trajectory["wf1"],
            trajectory["wf2"],
            c=trajectory["dataset_iteration"],
            cmap="magma_r",
            norm=norm,
            s=8,
            alpha=0.48,
            linewidths=0,
            zorder=2,
        )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="magma_r")
    colorbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="horizontal",
        location="top",
        fraction=0.08,
        pad=0.04,
    )
    colorbar.set_label("Training iteration")

    ax.axhline(0.0, color="black", linewidth=1.0, zorder=3)
    ax.axvline(0.0, color="black", linewidth=1.0, zorder=3)
    if show_feature_regions:
        add_feature_regions(ax)
    ax.set(
        xlabel=r"Weight from input 1, $w^{2,f}_{1,1}$",
        ylabel=r"Weight from input 2, $w^{2,f}_{1,2}$",
        xlim=(-1.06, 1.06),
        ylim=(-1.06, 1.06),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.24)

    for output_path in (Path(pdf_path), Path(svg_path)):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and plot Supplementary Fig. 3d."
    )
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--pdf-output", type=Path, default=PDF_PATH)
    parser.add_argument("--svg-output", type=Path, default=SVG_PATH)
    parser.add_argument("--grid-min", type=float, default=GRID_MIN)
    parser.add_argument("--grid-max", type=float, default=GRID_MAX)
    parser.add_argument("--grid-step", type=float, default=GRID_STEP)
    parser.add_argument("--initial-wf1", type=float)
    parser.add_argument("--initial-wf2", type=float)
    parser.add_argument("--initial-backward-weight", type=float, default=0.5)
    parser.add_argument("--theta", type=float, default=THETA)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--backward-learning-rate", type=float, default=0.005)
    parser.add_argument("--num-dataset-iterations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--tmax", type=float, default=DEFAULT_TMAX)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--output-delay-std", type=float, default=0.0)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-feature-regions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only and args.no_plot:
        raise ValueError("--plot-only and --no-plot cannot be used together")
    if not args.plot_only:
        if (args.initial_wf1 is None) != (args.initial_wf2 is None):
            raise ValueError(
                "--initial-wf1 and --initial-wf2 must be supplied together"
            )
        if args.initial_wf1 is None:
            values = grid_values(args.grid_min, args.grid_max, args.grid_step)
            initial_pairs = list(itertools.product(values.tolist(), repeat=2))
        else:
            initial_pairs = [(args.initial_wf1, args.initial_wf2)]
        row_count = write_grid_csv(
            args.results_path,
            initial_pairs,
            initial_backward_weight=args.initial_backward_weight,
            theta=args.theta,
            learning_rate=args.learning_rate,
            backward_learning_rate=args.backward_learning_rate,
            num_dataset_iterations=args.num_dataset_iterations,
            seed=args.seed,
            dt=args.dt,
            tmax=args.tmax,
            delay=args.delay,
            output_delay_std=args.output_delay_std,
        )
        print(
            f"Wrote {row_count} rows for {len(initial_pairs)} trajectories "
            f"to {args.results_path}"
        )
    if not args.no_plot:
        plot_results(
            args.results_path,
            pdf_path=args.pdf_output,
            svg_path=args.svg_output,
            show_feature_regions=not args.no_feature_regions,
        )
        print(f"Saved {args.pdf_output}")
        print(f"Saved {args.svg_output}")


if __name__ == "__main__":
    main()
