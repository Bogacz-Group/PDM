from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import spiking_xor  # noqa: E402
import supplementary_figure3d as figure3d  # noqa: E402


def test_regular_spike_train_matches_source_timing() -> None:
    spikes = spiking_xor.spike_train(5.0, 0.1, 50.0)
    np.testing.assert_array_equal(np.flatnonzero(spikes), [0, 100, 200, 300, 400])


def test_simulation_is_a_pure_weight_transition() -> None:
    rng = np.random.default_rng(3)
    forward_input, forward_output, backward = spiking_xor.initialize_weights(1, rng=rng)
    originals = tuple(
        array.copy() for array in (forward_input, forward_output, backward)
    )
    updated = spiking_xor.simulate(
        [0.0, 1.0],
        1.0,
        forward_input,
        forward_output,
        backward,
        True,
        0.4,
    )
    for actual, expected in zip((forward_input, forward_output, backward), originals):
        np.testing.assert_array_equal(actual, expected)
    assert updated[0].shape == (1, 2)
    # The source flattens this matrix after the first simulated example.
    assert updated[1].shape == (1,)
    assert updated[2].shape == (1, 1)
    assert all(np.isfinite(value) for value in updated[3:])


def test_trainable_smoke_and_determinism() -> None:
    config = {
        "wf11": -0.5,
        "wf12": 1.0,
        "wb": 0.5,
        "ITER": 1,
        "seed": 9,
    }
    first = figure3d.trainable(config)
    second = figure3d.trainable(config)
    assert first == second
    assert first["iterations"] == [1]
    assert first["sample_updates"] == 4
    assert len(first["wf1s"]) == len(first["wf2s"]) == len(first["wbs"]) == 1


def test_spiking_grid_is_inclusive() -> None:
    np.testing.assert_allclose(figure3d.grid_values(), np.arange(-1.0, 1.01, 0.25))
