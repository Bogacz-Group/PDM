"""Fast deterministic regression tests for Figure 6a-h."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from analytic import analytic_stdp_curve  # noqa: E402
from figure6 import _panel_colors, generate_results, panel_data, parse_args  # noqa: E402
from model import sigmoid_gate, stdp_curve, stdp_single  # noqa: E402


def test_gate_is_half_at_gamma() -> None:
    assert sigmoid_gate(0.02, plasticity_threshold=0.02) == 0.5


def test_historical_stdp_regression() -> None:
    np.testing.assert_allclose(
        stdp_single(-20.0), -0.019525457881115538, rtol=0, atol=1e-14
    )


def test_positive_delay_regression() -> None:
    np.testing.assert_allclose(
        stdp_single(10.0), 0.020908914804487422, rtol=0, atol=1e-14
    )


def test_historical_keyword_aliases() -> None:
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
    assert canonical == historical


def test_trace_and_percentage_conventions() -> None:
    _, simulation = stdp_single(-20.0, return_trace=True)
    np.testing.assert_array_equal(
        simulation.dendritic_error,
        simulation.output_signal[np.newaxis, :] - simulation.dendritic_potential,
    )
    absolute = stdp_curve([-20.0, 10.0], initial_weight=0.5)
    percent = stdp_curve([-20.0, 10.0], initial_weight=0.5, percent_change=True)
    np.testing.assert_allclose(percent, 200.0 * absolute, rtol=0, atol=1e-14)


def test_analytic_and_dispatcher_outputs() -> None:
    assert np.all(np.isfinite(analytic_stdp_curve([-50.0, 0.0, 49.0])))
    result = panel_data("stdp_curve", {"delays_ms": [-1.0, 0.0, 1.0]})
    assert result["delay_ms"] == [-1.0, 0.0, 1.0]
    assert all(isinstance(value, float) for value in result["weight_change_percent"])


def test_plot_palette_expands_for_custom_cli_series() -> None:
    assert len(_panel_colors("6d", 7)) == 7


def test_panel_6g_defaults_match_source_notebook() -> None:
    expected = (-0.5, 0.02, 0.05, 0.1, 0.2, 0.5)
    function_default = inspect.signature(generate_results).parameters[
        "gamma_values"
    ].default
    assert function_default == expected
    assert tuple(parse_args([]).gamma_values) == expected
