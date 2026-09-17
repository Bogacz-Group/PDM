"""Fast checks for Figure 7a."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

import figure7  # noqa: E402
import model  # noqa: E402


def test_historical_stdp_reference_value() -> None:
    change = model.stdp_single(10.0, initial_weight=0.5)
    assert change == pytest.approx(0.020908914804487422)


def test_figure7a_rows_cover_each_weight_and_delay() -> None:
    rows = figure7.generate_results(delays_ms=[-5.0, 5.0], initial_weights=[0.25, 0.75])
    assert len(rows) == 4
    assert {row["initial_weight"] for row in rows} == {0.25, 0.75}
    assert {row["delay_ms"] for row in rows} == {-5.0, 5.0}


def test_delay_grid_validation() -> None:
    with pytest.raises(ValueError):
        figure7._delay_grid(-10.0, 10.0, 0.0)
