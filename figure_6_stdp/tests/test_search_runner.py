"""Fast tests for the self-contained public mini-radas search runner."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from figure_6_stdp import run_search as runner  # noqa: E402


def test_profiles_and_run_ids_include_scientific_options() -> None:
    parser = runner.build_parser()
    full = parser.parse_args(["fig6i"])
    quick = parser.parse_args(["fig6j", "--quick"])
    assert runner.resolve_profile(full).num_samples == 1000
    assert runner.resolve_profile(quick).num_samples == 32
    legacy_id = runner.run_identifier(full, runner.resolve_profile(full))
    manuscript = parser.parse_args(["fig6i", "--weighting", "manuscript"])
    manuscript_id = runner.run_identifier(
        manuscript, runner.resolve_profile(manuscript)
    )
    assert legacy_id != manuscript_id


@pytest.mark.parametrize(
    "value", ["", ".", "..", "../escape", "folder/name", r"folder\name", "a b"]
)
def test_storage_names_reject_unsafe_components(value: str) -> None:
    with pytest.raises(ValueError):
        runner.validate_storage_component(value, "--name")


def test_fit_trial_table_reads_ray_config_columns() -> None:
    row = {
        "objective": 3.0,
        **{
            f"config/{name}": value
            for name, value in runner.fit_stdp.EMBEDDED_BEST_CONFIGS["bi2002"].items()
        },
    }
    trials = runner.fit_trials_frame(pd.DataFrame([row]))
    assert trials.loc[0, "objective"] == 3.0
    assert trials.loc[0, "initial_weight"] == row["config/initial_weight"]


def test_run_calls_public_api_and_writes_metadata(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    async def fake_run_experiment(**kwargs):
        calls.update(kwargs)
        return {"df": pd.DataFrame([{"objective": 1.0}])}

    fake_module = types.ModuleType("radas")
    fake_module.run_experiment = fake_run_experiment
    monkeypatch.setitem(sys.modules, "radas", fake_module)
    monkeypatch.setattr(runner, "validate_mini_radas", lambda *_: "verified-sha")
    monkeypatch.setattr(runner, "build_search", lambda *_: (object(), {}))

    def fake_materialize(_args, _profile, _frame, output):
        path = output / "result.csv"
        path.write_text("value\n1\n", encoding="utf-8")
        return {"result": path}

    monkeypatch.setattr(runner, "materialize_fit", fake_materialize)
    args = runner.build_parser().parse_args(
        [
            "fig6i",
            "--smoke",
            "--storage",
            str(tmp_path / "storage"),
            "--output",
            str(tmp_path / "output"),
            "--no-plots",
        ]
    )
    paths = asyncio.run(runner.run(args))
    metadata = json.loads(paths["run_metadata"].read_text(encoding="utf-8"))
    assert calls["run_with"] == "local"
    assert calls["dos"] == ["run", "analyze"]
    assert calls["trainable"] is runner.fit_stdp.stdp_fit_trainable
    assert metadata["orchestrator"]["verified_source_sha256"] == "verified-sha"
    assert metadata["completed_trials"] == 1


def test_source_validation_rejects_tampered_package(tmp_path: Path) -> None:
    package = tmp_path / "radas"
    package.mkdir()
    for filename in runner.MINI_RADAS_SOURCE_FILES:
        (package / filename).write_text("# tampered\n", encoding="utf-8")
    fake_module = types.SimpleNamespace(__file__=str(package / "__init__.py"))

    def fake_run_experiment(user_name, resources, run_with, local_storage_path, dos):
        del user_name, resources, run_with, local_storage_path, dos

    with pytest.raises(RuntimeError, match="do not match mini-radas commit"):
        runner.validate_mini_radas(fake_module, fake_run_experiment)
