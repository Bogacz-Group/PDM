"""Fast checks for the public fitting and figure-materialization code."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SPIKING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPIKING_DIR))

import fit_initial_strength  # noqa: E402
import fit_stdp  # noqa: E402


def test_bi2002_transform_and_woodin_units() -> None:
    bi = fit_stdp.load_dataset("bi2002")
    woodin = fit_stdp.load_dataset("woodin2003")
    assert len(bi["delays_ms"]) == 45
    assert bi["delta_weight_percent"][0] == pytest.approx(0.0)
    assert bi["delta_weight_percent"][1] == pytest.approx(-12.35955056179776)
    assert len(woodin["delays_ms"]) == 73
    assert woodin["delta_weight_percent"][0] == pytest.approx(-18.78787878787881)


def test_legacy_weighting_preserves_first_duplicate_window() -> None:
    delays = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    changes = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    weights = fit_stdp.legacy_variance_weights(
        delays, changes, half_window_observations=2
    )
    # Both delay-zero observations use list.index(0.0) == 0 and therefore the
    # same [0:2) legacy window, whose population variance is 1.
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(1.0)


def test_manuscript_weighting_uses_ms_window_and_declared_floor() -> None:
    weights = fit_stdp.manuscript_variance_weights(
        delays_ms=[0.0, 2.0, 10.0],
        delta_weight_percent=[0.0, 2.0, 3.0],
        half_width_ms=5.0,
        variance_floor=0.25,
    )
    assert weights.tolist() == pytest.approx([1.0, 1.0, 4.0])


def test_stdp_trainable_accepts_flat_and_nested_parameters(monkeypatch) -> None:
    def fake_stdp_single(
        *, delay_ms: float, initial_weight: float, **_: float
    ) -> float:
        return initial_weight * (delay_ms + 110.0) / 100.0

    monkeypatch.setattr(fit_stdp, "stdp_single", fake_stdp_single)
    params = {
        "transmission_delay_ms": 3.0,
        "decay_rate": 0.1,
        "plasticity_threshold": 0.2,
        "learning_rate": 0.3,
        "initial_weight": 0.7,
    }
    flat = fit_stdp.stdp_fit_trainable(
        {"dataset": "bi2002", "weighting": "none", **params}
    )
    nested = fit_stdp.stdp_fit_trainable(
        {"dataset": "bi2002", "weighting": "none", "parameters": params}
    )
    assert flat == pytest.approx(nested)
    assert np.isfinite(flat["objective"])


def test_initial_weight_mapping_hits_both_endpoints() -> None:
    experimental = np.asarray([10.0, 20.0, 40.0])
    slope, intercept = fit_initial_strength.solve_initial_weight_mapping(
        experimental,
        model_initial_weight_start=0.1,
        model_initial_weight_range=0.9,
    )
    mapped = experimental * slope + intercept
    assert mapped.min() == pytest.approx(0.1)
    assert mapped.max() == pytest.approx(1.0)


def test_initial_strength_trainable_is_pure_and_accepts_nested(monkeypatch) -> None:
    def fake_stdp_single(
        *, delay_ms: float, initial_weight: float, **_: float
    ) -> float:
        return initial_weight * delay_ms / 100.0

    monkeypatch.setattr(fit_initial_strength, "stdp_single", fake_stdp_single)
    nested = fit_initial_strength.initial_strength_fit_trainable(
        {"parameters": fit_initial_strength.EMBEDDED_BEST_CONFIG}
    )
    assert set(nested) == {
        "objective",
        "rmse",
        "mapping_slope_model_units_per_pa",
        "mapping_intercept_model_units",
    }
    assert np.isfinite(nested["objective"])


def test_provenance_json_matches_embedded_parameters() -> None:
    for dataset, filename in {
        "bi2002": "fig6i_bi2002.json",
        "woodin2003": "fig6j_woodin2003.json",
    }.items():
        payload = json.loads((SPIKING_DIR / "configs" / filename).read_text())
        assert payload["parameters"] == fit_stdp.EMBEDDED_BEST_CONFIGS[dataset]
    payload = json.loads(
        (SPIKING_DIR / "configs" / "fig7b_initial_strength.json").read_text()
    )
    assert payload["parameters"] == fit_initial_strength.EMBEDDED_BEST_CONFIG


def test_multi_dataset_fit_records_each_effective_seed(monkeypatch) -> None:
    fit_seeds = []
    recorded_seeds = []

    def fake_fit(*, seed, **_):
        fit_seeds.append(seed)
        return fit_stdp.EMBEDDED_BEST_CONFIGS["bi2002"].copy(), object()

    def fake_materialize(*, seed, **_):
        recorded_seeds.append(seed)
        return {}

    monkeypatch.setattr(fit_stdp, "run_optuna_fit", fake_fit)
    monkeypatch.setattr(fit_stdp, "materialize_result", fake_materialize)

    assert fit_stdp.main(["--fit", "--dataset", "all", "--seed", "13"]) == 0
    assert fit_seeds == [13, 14]
    assert recorded_seeds == [13, 14]


def test_embedded_initial_strength_rejects_manuscript_search_space() -> None:
    with pytest.raises(SystemExit) as error:
        fit_initial_strength.main(["--manuscript-search-space"])
    assert error.value.code == 2
