from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import weight_trajectories as wt  # noqa: E402


class WeightTrajectoryTests(unittest.TestCase):
    def test_default_grid_is_inclusive(self) -> None:
        values = wt.grid_values()
        self.assertEqual(len(values), 9)
        np.testing.assert_allclose(values, np.arange(-1.0, 1.01, 0.25))

    def test_one_batch_update_matches_source_equations(self) -> None:
        weights, backward_weight = wt.firing_rate_update([-0.5, 1.0], 0.5)
        np.testing.assert_allclose(weights, [-0.525, 0.95], atol=1e-12)
        self.assertTrue(np.isclose(backward_weight, 0.525))

    def test_trainable_records_paper_update_accounting(self) -> None:
        result = wt.trainable(
            {
                "wf11": -1.0,
                "wf12": -1.0,
                "wb": 0.5,
                "ITER": 2,
                "seed": 7,
            }
        )
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["sample_updates"], 8)
        self.assertEqual(result["iterations"], [1, 2])
        self.assertEqual(result["wf1s"], [-1.0, -1.0])
        self.assertEqual(result["wf2s"], [-1.0, -1.0])

    def test_trainable_is_deterministic(self) -> None:
        config = {"wf11": -0.5, "wf12": 1.0, "ITER": 3, "seed": 11}
        self.assertEqual(wt.trainable(config), wt.trainable(config))


if __name__ == "__main__":
    unittest.main()
