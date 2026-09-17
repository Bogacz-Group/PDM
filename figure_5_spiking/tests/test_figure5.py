"""Regression tests for the self-contained Figure 5 reproduction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from figure5 import generate_results, simulate_figure5  # noqa: E402
from model import sigmoid_gate  # noqa: E402


class Figure5Tests(unittest.TestCase):
    def test_gate_is_half_at_threshold(self) -> None:
        self.assertEqual(sigmoid_gate(0.02, plasticity_threshold=0.02), 0.5)

    def test_baseline_regression_and_error_sign(self) -> None:
        simulation = simulate_figure5(duration_ms=60.0)
        np.testing.assert_array_equal(
            simulation.dendritic_error,
            simulation.output_signal[np.newaxis, :]
            - simulation.dendritic_potential,
        )
        self.assertAlmostEqual(
            simulation.weights[0, -1], 0.34047292607759017, places=14
        )

    def test_default_result_shape_and_parameter_sets(self) -> None:
        historical = generate_results()
        manuscript = generate_results(parameter_set="manuscript")
        self.assertEqual(len(historical), 14_080)
        self.assertEqual(len(manuscript), 14_080)
        historical_alpha = {
            float(row["alpha"])
            for row in historical
            if row["section"] == "baseline"
        }
        manuscript_alpha = {
            float(row["alpha"])
            for row in manuscript
            if row["section"] == "baseline"
        }
        self.assertEqual(historical_alpha, {0.2})
        self.assertEqual(manuscript_alpha, {0.1})


if __name__ == "__main__":
    unittest.main()
