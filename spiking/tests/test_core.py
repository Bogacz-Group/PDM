"""Fast deterministic regression tests for the spiking figure core."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SPIKING_DIR = Path(__file__).resolve().parents[1]
if str(SPIKING_DIR) not in sys.path:
    sys.path.insert(0, str(SPIKING_DIR))

from analytic import analytic_stdp_curve  # noqa: E402
from figure5 import _series_colors  # noqa: E402
from figure6 import _panel_colors, panel_data  # noqa: E402
from model import sigmoid_gate, stdp_curve, stdp_single  # noqa: E402


class SpikingCoreTests(unittest.TestCase):
    def test_gate_is_half_at_gamma(self) -> None:
        self.assertEqual(sigmoid_gate(0.02, plasticity_threshold=0.02), 0.5)

    def test_historical_stdp_regression(self) -> None:
        self.assertAlmostEqual(stdp_single(-20.0), -0.019525457881115538, places=14)
        self.assertAlmostEqual(stdp_single(10.0), 0.020908914804487422, places=14)

    def test_historical_keyword_aliases(self) -> None:
        canonical = stdp_single(
            10.0,
            transmission_delay_ms=5.0,
            decay_rate=0.1,
            plasticity_threshold=0.02,
            learning_rate=0.01,
            initial_weight=0.5,
        )
        historical = stdp_single(
            10.0,
            syndelay="5 ms",
            decay="10 ms",
            epsilon="0.02",
            lrate="0.01",
            initialw="0.5",
        )
        self.assertEqual(canonical, historical)

    def test_trace_error_uses_manuscript_sign(self) -> None:
        _, simulation = stdp_single(-20.0, return_trace=True)
        np.testing.assert_array_equal(
            simulation.dendritic_error,
            simulation.output_signal[np.newaxis, :] - simulation.dendritic_potential,
        )

    def test_curve_percent_normalization(self) -> None:
        absolute = stdp_curve([-20.0, 10.0], initial_weight=0.5)
        percent = stdp_curve([-20.0, 10.0], initial_weight=0.5, percent_change=True)
        np.testing.assert_allclose(percent, 200.0 * absolute, rtol=0, atol=1e-14)

    def test_analytic_approximation_is_finite(self) -> None:
        curve = analytic_stdp_curve([-50.0, 0.0, 49.0])
        self.assertTrue(np.all(np.isfinite(curve)))

    def test_panel_dispatcher_returns_serializable_lists(self) -> None:
        result = panel_data("stdp_curve", {"delays_ms": [-1.0, 0.0, 1.0]})
        self.assertEqual(result["delay_ms"], [-1.0, 0.0, 1.0])
        self.assertTrue(
            all(isinstance(value, float) for value in result["weight_change_percent"])
        )

    def test_plot_palettes_expand_for_custom_cli_series(self) -> None:
        self.assertEqual(len(_series_colors(["#000000"], 5)), 5)
        self.assertEqual(len(_panel_colors("6d", 7)), 7)


if __name__ == "__main__":
    unittest.main()
