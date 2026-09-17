"""Fast checks for the Figure 7b fit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

import fit_initial_strength as fit  # noqa: E402


def test_dataset_and_recorded_objective() -> None:
    data = fit.load_dataset()
    assert len(data["initial_epsp_pa"]) == 37
    result = fit.evaluate_config(data, fit.EMBEDDED_BEST_CONFIG)
    assert result["objective"] == pytest.approx(13017.619425355382)


def test_mapping_hits_declared_endpoints() -> None:
    values = np.asarray([10.0, 20.0, 40.0])
    slope, intercept = fit.solve_initial_weight_mapping(values, 0.1, 0.9)
    mapped = values * slope + intercept
    assert mapped.min() == pytest.approx(0.1)
    assert mapped.max() == pytest.approx(1.0)


def test_config_snapshot_matches_embedded_parameters() -> None:
    payload = json.loads(
        (EXPERIMENT_DIR / "configs" / "fig7b_initial_strength.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["parameters"] == fit.EMBEDDED_BEST_CONFIG


def test_search_space_variants_preserve_documented_difference() -> None:
    assert fit.search_bounds("legacy")["model_initial_weight_range"][:2] == (0.5, 2.0)
    assert fit.search_bounds("manuscript")["model_initial_weight_range"][:2] == (
        0.0,
        0.3,
    )


def test_trainable_accepts_nested_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fit,
        "stdp_single",
        lambda *, delay_ms, initial_weight, **_: initial_weight * delay_ms / 100.0,
    )
    result = fit.initial_strength_fit_trainable(
        {"parameters": fit.EMBEDDED_BEST_CONFIG}
    )
    assert np.isfinite(result["objective"])
    assert set(result) == {
        "objective",
        "rmse",
        "mapping_slope_model_units_per_pa",
        "mapping_intercept_model_units",
    }
