from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import weight_trajectories as wt  # noqa: E402


def test_default_grid_is_inclusive() -> None:
    values = wt.grid_values()
    assert len(values) == 9
    np.testing.assert_allclose(values, np.arange(-1.0, 1.01, 0.25))


def test_one_batch_update_matches_source_equations() -> None:
    weights, backward_weight = wt.firing_rate_update([-0.5, 1.0], 0.5)
    np.testing.assert_allclose(weights, [-0.525, 0.95], atol=1e-12)
    assert np.isclose(backward_weight, 0.525)


def test_trainable_records_paper_update_accounting() -> None:
    result = wt.trainable(
        {
            "wf11": -1.0,
            "wf12": -1.0,
            "wb": 0.5,
            "ITER": 2,
            "seed": 7,
        }
    )
    assert result["optimizer_steps"] == 2
    assert result["sample_updates"] == 8
    assert result["iterations"] == [1, 2]
    assert result["wf1s"] == [-1.0, -1.0]
    assert result["wf2s"] == [-1.0, -1.0]


def test_trainable_is_deterministic() -> None:
    config = {"wf11": -0.5, "wf12": 1.0, "ITER": 3, "seed": 11}
    assert wt.trainable(config) == wt.trainable(config)
