"""Fast checks for the dedicated public mini-radas runner."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

import fit_initial_strength as fit  # noqa: E402
import run_with_radas as runner  # noqa: E402


def test_profiles_and_run_ids_include_scientific_options() -> None:
    parser = runner.build_parser()
    full = parser.parse_args([])
    quick = parser.parse_args(["--quick"])
    manuscript = parser.parse_args(["--smoke", "--search-space", "manuscript"])
    assert runner.sample_count(full) == ("full", 1000)
    assert runner.sample_count(quick) == ("quick", 32)
    profile, count = runner.sample_count(manuscript)
    assert runner.run_identifier(manuscript, profile, count) != runner.run_identifier(
        parser.parse_args(["--smoke"]), "smoke", 1
    )


@pytest.mark.parametrize("value", ["", ".", "..", "../x", "a/b", r"a\b", "a b"])
def test_unsafe_storage_components_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        runner.validate_component(value, "--name")


def test_trial_table_reads_ray_config_columns() -> None:
    row = {
        "objective": 3.0,
        **{f"config/{name}": value for name, value in fit.EMBEDDED_BEST_CONFIG.items()},
    }
    trials = runner.fit_trials_frame(pd.DataFrame([row]))
    assert trials.loc[0, "objective"] == 3.0
    assert trials.loc[0, "model_initial_weight_range"] == pytest.approx(
        fit.EMBEDDED_BEST_CONFIG["model_initial_weight_range"]
    )


def test_source_validation_rejects_tampered_package(tmp_path: Path) -> None:
    package = tmp_path / "radas"
    package.mkdir()
    for filename in runner.MINI_RADAS_SOURCE_FILES:
        (package / filename).write_text("# tampered\n", encoding="utf-8")
    module = types.SimpleNamespace(__file__=str(package / "__init__.py"))

    def fake(user_name, resources, run_with, local_storage_path, dos):
        del user_name, resources, run_with, local_storage_path, dos

    with pytest.raises(RuntimeError, match="do not match public mini-radas"):
        runner.validate_mini_radas(module, fake)


def test_run_calls_public_local_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    async def fake_run_experiment(**kwargs):
        calls.update(kwargs)
        row = {"objective": 1.0, **fit.EMBEDDED_BEST_CONFIG}
        return {"df": pd.DataFrame([row])}

    module = types.ModuleType("radas")
    module.run_experiment = fake_run_experiment
    monkeypatch.setitem(sys.modules, "radas", module)
    monkeypatch.setattr(runner, "validate_mini_radas", lambda *_: "verified-hash")
    monkeypatch.setattr(runner, "build_search", lambda _args, _count: ({}, object()))
    artifact = tmp_path / "best.json"
    artifact.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        runner, "materialize", lambda *_args, **_kwargs: {"json": artifact}
    )
    args = runner.build_parser().parse_args(
        [
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
    assert metadata["orchestrator"]["verified_source_sha256"] == "verified-hash"
    assert metadata["completed_trials"] == 1
