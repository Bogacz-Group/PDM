from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import spiking_xor  # noqa: E402
import supplementary_figure3d as figure3d  # noqa: E402


class SupplementaryFigure3dTests(unittest.TestCase):
    def test_regular_spike_train_matches_source_timing(self) -> None:
        spikes = spiking_xor.spike_train(5.0, 0.1, 50.0)
        np.testing.assert_array_equal(
            np.flatnonzero(spikes), [0, 100, 200, 300, 400]
        )

    def test_simulation_is_a_pure_weight_transition(self) -> None:
        rng = np.random.default_rng(3)
        forward_input, forward_output, backward = spiking_xor.initialize_weights(
            1, rng=rng
        )
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
        for actual, expected in zip(
            (forward_input, forward_output, backward), originals
        ):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(updated[0].shape, (1, 2))
        # The source flattens this matrix after the first simulated example.
        self.assertEqual(updated[1].shape, (1,))
        self.assertEqual(updated[2].shape, (1, 1))
        self.assertTrue(all(np.isfinite(value) for value in updated[3:]))

    def test_trainable_smoke_and_determinism(self) -> None:
        config = {
            "wf11": -0.5,
            "wf12": 1.0,
            "wb": 0.5,
            "ITER": 1,
            "seed": 9,
        }
        first = figure3d.trainable(config)
        second = figure3d.trainable(config)
        self.assertEqual(first, second)
        self.assertEqual(first["iterations"], [1])
        self.assertEqual(first["sample_updates"], 4)
        self.assertEqual(len(first["wf1s"]), 1)
        self.assertEqual(len(first["wf2s"]), 1)
        self.assertEqual(len(first["wbs"]), 1)

    def test_spiking_grid_is_inclusive(self) -> None:
        np.testing.assert_allclose(
            figure3d.grid_values(), np.arange(-1.0, 1.01, 0.25)
        )


if __name__ == "__main__":
    unittest.main()
