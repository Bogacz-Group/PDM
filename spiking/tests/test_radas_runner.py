"""Fast tests for the publication-facing mini-radas runner."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from spiking import run_with_radas as runner  # noqa: E402


def _rate_result() -> dict[str, object]:
    return {
        "panel": "supplementary_figure3c",
        "model": "firing_rate",
        "seed": 7,
        "initial_wf1": -0.5,
        "initial_wf2": 1.0,
        "initial_wb": 0.5,
        "theta_forward": 0.5,
        "theta_backward": 0.5,
        "learning_rate": 0.1,
        "backward_learning_rate": 0.1,
        "num_dataset_iterations": 2,
        "optimizer_steps": 2,
        "sample_updates": 8,
        "iterations": [1, 2],
        "wf1s": [-0.51, -0.52],
        "wf2s": [0.98, 0.96],
        "wbs": [0.51, 0.52],
        "final_wf1": -0.52,
        "final_wf2": 0.96,
        "final_wb": 0.52,
        "weight_displacement": 0.04472135955,
    }


def test_profiles_have_publication_and_fast_defaults() -> None:
    parser = runner.build_parser()
    full_grid = runner.resolve_profile(parser.parse_args(["supp3c"]))
    quick_grid = runner.resolve_profile(parser.parse_args(["supp3d", "--quick"]))
    smoke_fit = runner.resolve_profile(parser.parse_args(["fig6i", "--smoke"]))
    assert (full_grid.iterations, full_grid.grid_step) == (128, 0.25)
    assert (quick_grid.iterations, quick_grid.grid_step) == (4, 1.0)
    assert smoke_fit.num_samples == 1


def test_relative_paths_are_resolved_from_script() -> None:
    assert (
        runner.resolve_script_path("somewhere/results")
        == (runner.HERE / "somewhere" / "results").resolve()
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "..",
        "../escape",
        "folder/name",
        r"folder\name",
        "name with spaces",
        "name;command",
        ".hidden",
    ],
)
def test_storage_components_reject_unsafe_names(value: str) -> None:
    with pytest.raises(ValueError):
        runner.validate_storage_component(value, "--test-name")


@pytest.mark.parametrize("value", ["pdm-public", "paper_run-1", "run.v2"])
def test_storage_components_accept_conservative_names(value: str) -> None:
    assert runner.validate_storage_component(value, "--test-name") == value


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--experiment-name", "."),
        ("--experiment-name", ".."),
        ("--experiment-name", "../escape"),
        ("--experiment-name", "folder/name"),
        ("--user-name", "."),
        ("--user-name", ".."),
        ("--user-name", "../escape"),
        ("--user-name", r"folder\name"),
    ],
)
def test_unsafe_storage_names_fail_before_creating_directories(
    option: str, value: str, tmp_path: Path
) -> None:
    storage = tmp_path / "must-not-exist"
    output = tmp_path / "output-must-not-exist"
    args = runner.build_parser().parse_args(
        [
            "supp3c",
            "--smoke",
            option,
            value,
            "--storage",
            str(storage),
            "--output",
            str(output),
            "--no-plots",
        ]
    )
    with pytest.raises(ValueError):
        __import__("asyncio").run(runner.run(args))
    assert not storage.exists()
    assert not output.exists()


def test_grid_materialization_explodes_list_metrics(tmp_path: Path) -> None:
    paths = runner._materialize_grid(
        "supp3c",
        pd.DataFrame([_rate_result()]),
        tmp_path,
        plots=False,
    )
    trajectories = pd.read_csv(paths["trajectories_csv"])
    trials = pd.read_csv(paths["trials_csv"])
    assert trajectories["dataset_iteration"].tolist() == [1, 2]
    assert trajectories["sample_updates"].tolist() == [4, 8]
    assert trials.loc[0, "final_wf2"] == 0.96


def test_fit_trial_table_reads_ray_config_columns() -> None:
    row = {
        "objective": 3.0,
        "unweighted_rmse": 2.0,
        **{
            f"config/{name}": value
            for name, value in runner.fit_stdp.EMBEDDED_BEST_CONFIGS["bi2002"].items()
        },
    }
    trials = runner._fit_trials_frame("fig6i", pd.DataFrame([row]))
    assert trials.loc[0, "objective"] == 3.0
    assert trials.loc[0, "initial_weight"] == row["config/initial_weight"]


def test_run_calls_public_local_api_and_writes_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    async def fake_run_experiment(**kwargs):
        calls.update(kwargs)
        return {"df": pd.DataFrame([_rate_result()]), "tuner": object()}

    fake_module = types.ModuleType("radas")
    fake_module.run_experiment = fake_run_experiment
    monkeypatch.setitem(sys.modules, "radas", fake_module)
    monkeypatch.setattr(runner, "validate_mini_radas", lambda *_args: "test-sha256")
    monkeypatch.setattr(
        runner,
        "build_experiment_spec",
        lambda _args, _profile: runner.ExperimentSpec(
            runner.firing_rate_trajectory_trainable,
            {},
            object(),
            None,
            None,
        ),
    )
    args = runner.build_parser().parse_args(
        [
            "supp3c",
            "--smoke",
            "--seed",
            "7",
            "--storage",
            str(tmp_path / "storage"),
            "--output",
            str(tmp_path / "output"),
            "--no-plots",
        ]
    )
    paths = __import__("asyncio").run(runner.run(args))
    metadata = json.loads(paths["run_metadata"].read_text(encoding="utf-8"))
    assert calls["run_with"] == "local"
    assert calls["dos"] == ["run", "analyze"]
    assert calls["user_name"] == "pdm-public"
    assert metadata["orchestrator"] == {
        "package": "mini-radas",
        "expected_commit": runner.MINI_RADAS_COMMIT,
        "verified_source_sha256": "test-sha256",
        "api": "radas.run_experiment",
    }
    assert metadata["completed_trials"] == 1


def test_default_run_identifier_changes_with_scientific_options() -> None:
    parser = runner.build_parser()
    legacy = parser.parse_args(["fig6i", "--smoke", "--weighting", "legacy"])
    manuscript = parser.parse_args(["fig6i", "--smoke", "--weighting", "manuscript"])
    legacy_id = runner.run_identifier(legacy, runner.resolve_profile(legacy))
    manuscript_id = runner.run_identifier(
        manuscript, runner.resolve_profile(manuscript)
    )
    assert legacy_id != manuscript_id
    assert legacy_id.startswith("smoke-seed-0-")


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
