"""Fast checks for the Figure 6i-j fitting code and records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import fit_stdp  # noqa: E402


def test_dataset_transforms() -> None:
    bi = fit_stdp.load_dataset("bi2002")
    woodin = fit_stdp.load_dataset("woodin2003")
    assert len(bi["delays_ms"]) == 45
    assert bi["delta_weight_percent"][1] == pytest.approx(-12.35955056179776)
    assert len(woodin["delays_ms"]) == 73
    assert woodin["delta_weight_percent"][0] == pytest.approx(-18.78787878787881)


def test_legacy_duplicate_and_manuscript_window_weighting() -> None:
    weights = fit_stdp.legacy_variance_weights(
        [0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        half_window_observations=2,
    )
    assert weights[:2].tolist() == pytest.approx([1.0, 1.0])
    manuscript = fit_stdp.manuscript_variance_weights(
        delays_ms=[0.0, 2.0, 10.0],
        delta_weight_percent=[0.0, 2.0, 3.0],
        half_width_ms=5.0,
        variance_floor=0.25,
    )
    assert manuscript.tolist() == pytest.approx([1.0, 1.0, 4.0])


def test_trainable_accepts_flat_and_nested_parameters(monkeypatch) -> None:
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


def test_recorded_configs_match_embedded_parameters() -> None:
    for dataset, filename in {
        "bi2002": "fig6i_bi2002.json",
        "woodin2003": "fig6j_woodin2003.json",
    }.items():
        payload = json.loads((EXPERIMENT_DIR / "configs" / filename).read_text())
        assert payload["parameters"] == fit_stdp.EMBEDDED_BEST_CONFIGS[dataset]


def test_multi_dataset_fit_records_each_effective_seed(monkeypatch) -> None:
    fit_seeds: list[int] = []
    recorded_seeds: list[int] = []

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
