#!/usr/bin/env python3
"""Reproduce Supplementary Fig. 3c firing-rate weight trajectories.

The original notebook ran one hidden ReLU unit on the complete four-example
XOR batch.  Each dataset iteration therefore makes one vectorised weight
update which is the sum of four sample contributions.  The manuscript refers
to the resulting 128 x 4 = 512 sample-level updates.

This file contains the minimal one-hidden-unit dynamics used by that notebook
rather than depending on the experiment framework.  In particular, the
input-to-hidden and output-to-hidden weights learn while the other two weight
matrices (which cannot affect the plotted trajectory) are frozen.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "results" / "supplementary_figure3c.csv"

GRID_MIN = -1.0
GRID_MAX = 1.0
GRID_STEP = 0.25
LEARNING_RATE = 0.1
INITIAL_BACKWARD_WEIGHT = 0.5
THETA_FORWARD = 0.5
THETA_BACKWARD = 0.5
NUM_DATASET_ITERATIONS = 128
SEED = 0

XOR_INPUTS = np.asarray(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    dtype=np.float64,
)
XOR_TARGETS = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64)

CSV_COLUMNS = (
    "panel",
    "model",
    "seed",
    "initial_wf1",
    "initial_wf2",
    "initial_wb",
    "theta_forward",
    "theta_backward",
    "learning_rate",
    "backward_learning_rate",
    "dataset_iteration",
    "optimizer_step",
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


def _validate_dynamics(
    *,
    learning_rate: float,
    backward_learning_rate: float,
    theta_forward: float,
    theta_backward: float,
    num_dataset_iterations: int,
) -> None:
    if learning_rate < 0.0 or backward_learning_rate < 0.0:
        raise ValueError("learning rates must be non-negative")
    if theta_forward < 0.0 or theta_backward < 0.0:
        raise ValueError("theta coefficients must be non-negative")
    if theta_forward + theta_backward <= 0.0:
        raise ValueError("at least one theta coefficient must be positive")
    if num_dataset_iterations < 0:
        raise ValueError("num_dataset_iterations must be non-negative")


def firing_rate_update(
    forward_weights: Sequence[float],
    backward_weight: float,
    *,
    learning_rate: float = LEARNING_RATE,
    backward_learning_rate: float = LEARNING_RATE,
    theta_forward: float = THETA_FORWARD,
    theta_backward: float = THETA_BACKWARD,
) -> tuple[np.ndarray, float]:
    """Apply one complete-XOR-batch update from the source model.

    The hidden activity is the ReLU of the weighted mean of forward and
    backward dendritic predictions.  Plasticity is gated by positive hidden
    activity, exactly as in the historical ``ThreeQModel`` path.
    """

    _validate_dynamics(
        learning_rate=float(learning_rate),
        backward_learning_rate=float(backward_learning_rate),
        theta_forward=float(theta_forward),
        theta_backward=float(theta_backward),
        num_dataset_iterations=1,
    )
    weights = np.asarray(forward_weights, dtype=np.float64).copy()
    if weights.shape != (2,):
        raise ValueError("forward_weights must contain exactly two values")

    backward_weight = float(backward_weight)
    theta_sum = float(theta_forward + theta_backward)
    forward_potential = XOR_INPUTS @ weights
    backward_potential = XOR_TARGETS * backward_weight
    hidden_potential = (
        theta_forward * forward_potential + theta_backward * backward_potential
    ) / theta_sum
    hidden_activity = np.maximum(hidden_potential, 0.0)
    plasticity_gate = (hidden_activity > 0.0).astype(np.float64)

    forward_error = (
        2.0 * theta_forward * (hidden_activity - forward_potential) * plasticity_gate
    )
    backward_error = (
        2.0 * theta_backward * (hidden_activity - backward_potential) * plasticity_gate
    )

    weights += float(learning_rate) * (XOR_INPUTS.T @ forward_error)
    backward_weight += float(backward_learning_rate) * float(
        XOR_TARGETS @ backward_error
    )
    return weights, backward_weight


def run_trajectory(
    initial_wf1: float,
    initial_wf2: float,
    *,
    initial_backward_weight: float = INITIAL_BACKWARD_WEIGHT,
    learning_rate: float = LEARNING_RATE,
    backward_learning_rate: float | None = None,
    theta_forward: float = THETA_FORWARD,
    theta_backward: float = THETA_BACKWARD,
    num_dataset_iterations: int = NUM_DATASET_ITERATIONS,
    seed: int = SEED,
) -> dict[str, object]:
    """Train one initial condition and return scalar and list-valued fields."""

    if backward_learning_rate is None:
        backward_learning_rate = learning_rate
    _validate_dynamics(
        learning_rate=float(learning_rate),
        backward_learning_rate=float(backward_learning_rate),
        theta_forward=float(theta_forward),
        theta_backward=float(theta_backward),
        num_dataset_iterations=int(num_dataset_iterations),
    )

    # The tracked dynamics contain no random operation.  Retaining an explicit
    # seed makes the trainable interface deterministic and records provenance.
    seed = int(seed)
    np.random.default_rng(seed)

    weights = np.asarray([initial_wf1, initial_wf2], dtype=np.float64)
    backward_weight = float(initial_backward_weight)
    iterations: list[int] = []
    wf1s: list[float] = []
    wf2s: list[float] = []
    wbs: list[float] = []

    for dataset_iteration in range(1, int(num_dataset_iterations) + 1):
        weights, backward_weight = firing_rate_update(
            weights,
            backward_weight,
            learning_rate=float(learning_rate),
            backward_learning_rate=float(backward_learning_rate),
            theta_forward=float(theta_forward),
            theta_backward=float(theta_backward),
        )
        iterations.append(dataset_iteration)
        wf1s.append(float(weights[0]))
        wf2s.append(float(weights[1]))
        wbs.append(float(backward_weight))

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
        "panel": "supplementary_figure3c",
        "model": "firing_rate",
        "seed": seed,
        "initial_wf1": float(initial_wf1),
        "initial_wf2": float(initial_wf2),
        "initial_wb": float(initial_backward_weight),
        "theta_forward": float(theta_forward),
        "theta_backward": float(theta_backward),
        "learning_rate": float(learning_rate),
        "backward_learning_rate": float(backward_learning_rate),
        "num_dataset_iterations": int(num_dataset_iterations),
        "optimizer_steps": int(num_dataset_iterations),
        "sample_updates": int(num_dataset_iterations) * len(XOR_TARGETS),
        "iterations": iterations,
        "wf1s": wf1s,
        "wf2s": wf2s,
        "wbs": wbs,
        "final_wf1": final_wf1,
        "final_wf2": final_wf2,
        "final_wb": final_wb,
        "weight_displacement": displacement,
    }


def firing_rate_trajectory_trainable(
    config: Mapping[str, object],
) -> dict[str, object]:
    """mini-radas-compatible pure trainable for one initial condition.

    Both descriptive option names and the historical notebook names
    (``wf11``, ``wf12``, ``wb``, ``ITER``) are accepted.
    """

    return run_trajectory(
        initial_wf1=float(config.get("initial_wf1", config.get("wf11", 0.1))),
        initial_wf2=float(config.get("initial_wf2", config.get("wf12", 0.1))),
        initial_backward_weight=float(
            config.get("initial_backward_weight", config.get("wb", 0.5))
        ),
        learning_rate=float(config.get("learning_rate", config.get("lr", 0.1))),
        backward_learning_rate=float(
            config.get(
                "backward_learning_rate",
                config.get(
                    "lr_w2b", config.get("learning_rate", config.get("lr", 0.1))
                ),
            )
        ),
        theta_forward=float(config.get("theta_forward", config.get("theta_f", 0.5))),
        theta_backward=float(config.get("theta_backward", config.get("theta_b", 0.5))),
        num_dataset_iterations=int(
            config.get("num_dataset_iterations", config.get("ITER", 128))
        ),
        seed=int(config.get("seed", SEED)),
    )


def trainable(config: Mapping[str, object]) -> dict[str, object]:
    """Backward-compatible short alias for ``firing_rate_trajectory_trainable``."""

    return firing_rate_trajectory_trainable(config)


def trajectory_rows(result: Mapping[str, object]) -> Iterator[dict[str, object]]:
    """Explode one list-valued trainable result into long-form CSV rows."""

    iterations = list(result["iterations"])
    wf1s = list(result["wf1s"])
    wf2s = list(result["wf2s"])
    wbs = list(result["wbs"])
    if not (len(iterations) == len(wf1s) == len(wf2s) == len(wbs)):
        raise ValueError("trajectory fields have inconsistent lengths")

    scalar_keys = CSV_COLUMNS[:10]
    for dataset_iteration, wf1, wf2, wb in zip(iterations, wf1s, wf2s, wbs):
        row = {key: result[key] for key in scalar_keys}
        row.update(
            {
                "dataset_iteration": int(dataset_iteration),
                "optimizer_step": int(dataset_iteration),
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
    """Run initial conditions sequentially and yield long-form rows."""

    for initial_wf1, initial_wf2 in initial_pairs:
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
    """Run a sequential grid and write the paper-ready long-form CSV."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Fig. 3c firing-rate trajectories."
    )
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--grid-min", type=float, default=GRID_MIN)
    parser.add_argument("--grid-max", type=float, default=GRID_MAX)
    parser.add_argument("--grid-step", type=float, default=GRID_STEP)
    parser.add_argument("--initial-wf1", type=float)
    parser.add_argument("--initial-wf2", type=float)
    parser.add_argument("--initial-backward-weight", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--backward-learning-rate", type=float, default=0.1)
    parser.add_argument("--theta-forward", type=float, default=0.5)
    parser.add_argument("--theta-backward", type=float, default=0.5)
    parser.add_argument("--num-dataset-iterations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.initial_wf1 is None) != (args.initial_wf2 is None):
        raise ValueError("--initial-wf1 and --initial-wf2 must be supplied together")
    if args.initial_wf1 is None:
        values = grid_values(args.grid_min, args.grid_max, args.grid_step)
        initial_pairs = list(itertools.product(values.tolist(), repeat=2))
    else:
        initial_pairs = [(args.initial_wf1, args.initial_wf2)]

    row_count = write_grid_csv(
        args.results_path,
        initial_pairs,
        initial_backward_weight=args.initial_backward_weight,
        learning_rate=args.learning_rate,
        backward_learning_rate=args.backward_learning_rate,
        theta_forward=args.theta_forward,
        theta_backward=args.theta_backward,
        num_dataset_iterations=args.num_dataset_iterations,
        seed=args.seed,
    )
    print(
        f"Wrote {row_count} rows for {len(initial_pairs)} trajectories "
        f"to {args.results_path}"
    )


if __name__ == "__main__":
    main()
