"""Pure unit tests for the Supplementary Figure 3d mini-radas runner."""

from __future__ import annotations

import asyncio
import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_with_radas as runner  # noqa: E402


def _result() -> dict[str, object]:
    return {
        "panel": "supplementary_figure3d",
        "model": "spiking",
        "seed": 7,
        "initial_wf1": -0.5,
        "initial_wf2": 1.0,
        "initial_wb": 0.5,
        "theta": 0.4,
        "learning_rate": 0.005,
        "backward_learning_rate": 0.005,
        "dt": 0.1,
        "tmax": 50.0,
        "delay": 5.0,
        "output_delay_std": 0.0,
        "decay_per_ms": 0.1,
        "spike_threshold": 0.5,
        "reset_potential": -0.5,
        "hidden_gate_epsilon": 0.05,
        "output_gate_epsilon": -1.0,
        "gate_gain": 200.0,
        "num_dataset_iterations": 2,
        "sample_updates": 8,
        "iterations": [1, 2],
        "wf1s": [-0.49, -0.48],
        "wf2s": [0.98, 0.96],
        "wbs": [0.51, 0.52],
        "final_wf1": -0.48,
        "final_wf2": 0.96,
        "final_wb": 0.52,
        "weight_displacement": 0.04472135955,
        "last_output": 0.0,
        "last_dendritic_error": 0.1,
    }


class ProfileTests(unittest.TestCase):
    def test_profiles_have_exact_workloads(self) -> None:
        parser = runner.build_parser()
        full = runner.resolve_profile(parser.parse_args([]))
        quick = runner.resolve_profile(parser.parse_args(["--quick"]))
        smoke = runner.resolve_profile(parser.parse_args(["--smoke"]))

        self.assertEqual((full.iterations, len(full.grid_values)), (64, 9))
        self.assertEqual(
            full.grid_values,
            (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0),
        )
        self.assertEqual(full.expected_trials, 81)
        self.assertEqual(full.grid_values[0], -1.0)
        self.assertEqual(full.grid_values[-1], 1.0)
        self.assertEqual((quick.iterations, quick.grid_values), (4, (-1.0, 0.0, 1.0)))
        self.assertEqual((smoke.iterations, smoke.grid_values), (1, (0.0,)))

    def test_experiment_spec_uses_fixed_model_parameters(self) -> None:
        class FakeTune:
            class TuneConfig:
                pass

            @staticmethod
            def grid_search(values):
                return ("grid", tuple(values))

        profile = runner.RunProfile("quick", 4, (-1.0, 0.0, 1.0))
        with mock.patch.object(runner, "_load_tune", return_value=FakeTune):
            spec = runner.build_experiment_spec(profile, seed=7)
        self.assertEqual(spec.param_space["initial_wf1"], ("grid", profile.grid_values))
        self.assertEqual(spec.param_space["initial_wf2"], ("grid", profile.grid_values))
        self.assertEqual(
            {
                name: spec.param_space[name]
                for name in (
                    "initial_backward_weight",
                    "theta",
                    "learning_rate",
                    "backward_learning_rate",
                    "dt",
                    "tmax",
                    "delay",
                    "output_delay_std",
                )
            },
            {
                "initial_backward_weight": 0.5,
                "theta": 0.4,
                "learning_rate": 0.005,
                "backward_learning_rate": 0.005,
                "dt": 0.1,
                "tmax": 50.0,
                "delay": 5.0,
                "output_delay_std": 0.0,
            },
        )
        self.assertEqual(spec.param_space["num_dataset_iterations"], 4)
        self.assertEqual(spec.param_space["seed"], 7)

    def test_relative_paths_are_script_relative(self) -> None:
        self.assertEqual(
            runner.resolve_script_path("somewhere/results"),
            (runner.HERE / "somewhere" / "results").resolve(),
        )


class StorageSafetyTests(unittest.TestCase):
    def test_rejects_unsafe_components(self) -> None:
        unsafe = (
            "",
            ".",
            "..",
            "../escape",
            "folder/name",
            r"folder\name",
            "name with spaces",
            "name;command",
            ".hidden",
        )
        for value in unsafe:
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.validate_storage_component(value, "--name")

    def test_accepts_conservative_components(self) -> None:
        for value in ("pdm-public", "paper_run-1", "run.v2"):
            with self.subTest(value=value):
                self.assertEqual(
                    runner.validate_storage_component(value, "--name"), value
                )


class FingerprintTests(unittest.TestCase):
    def test_validation_accepts_exact_bytes_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory) / "radas"
            package.mkdir()
            for index, filename in enumerate(runner.MINI_RADAS_SOURCE_FILES):
                (package / filename).write_text(
                    f"# fixture {index}\n", encoding="utf-8"
                )
            module = types.SimpleNamespace(__file__=str(package / "__init__.py"))

            def public_api(user_name, resources, run_with, local_storage_path, dos):
                del user_name, resources, run_with, local_storage_path, dos

            expected = runner._mini_radas_source_fingerprint(module)
            with mock.patch.object(runner, "MINI_RADAS_SOURCE_SHA256", expected):
                self.assertEqual(
                    runner.validate_mini_radas(module, public_api), expected
                )
                (package / "core.py").write_text("# tampered\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "do not match mini-radas"):
                    runner.validate_mini_radas(module, public_api)

    def test_validation_rejects_wrong_api(self) -> None:
        def incomplete_api(user_name):
            del user_name

        with self.assertRaisesRegex(RuntimeError, "missing parameters"):
            runner.validate_mini_radas(types.SimpleNamespace(), incomplete_api)


class MaterializationTests(unittest.TestCase):
    def test_materialization_writes_trajectory_and_trial_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            paths = runner.materialize_results(
                pd.DataFrame([_result()]), output, plots=False
            )

            with paths["trajectories_csv"].open(
                newline="", encoding="utf-8"
            ) as handle:
                trajectories = list(csv.DictReader(handle))
            with paths["trials_csv"].open(newline="", encoding="utf-8") as handle:
                trials = list(csv.DictReader(handle))

            self.assertEqual(
                [row["dataset_iteration"] for row in trajectories], ["1", "2"]
            )
            self.assertEqual(
                [row["sample_updates"] for row in trajectories], ["4", "8"]
            )
            self.assertEqual(len(trials), 1)
            self.assertEqual(float(trials[0]["final_wf2"]), 0.96)
            self.assertNotIn("pdf", paths)
            self.assertNotIn("svg", paths)

    def test_trial_count_must_match_grid(self) -> None:
        profile = runner.RunProfile("quick", 4, (-1.0, 0.0, 1.0))
        runner.validate_trial_count(pd.DataFrame(index=range(9)), profile)
        with self.assertRaisesRegex(RuntimeError, "returned 8 trials; expected 9"):
            runner.validate_trial_count(pd.DataFrame(index=range(8)), profile)

    def test_metadata_records_public_source_and_fixed_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            args = types.SimpleNamespace(seed=7, cpus_per_trial=1.0)
            profile = runner.RunProfile("smoke", 1, (0.0,))
            metadata = runner._write_metadata(
                args=args,
                profile=profile,
                frame=pd.DataFrame([_result()]),
                output=output,
                storage=output / "storage",
                experiment_name="pdm-supp3d-smoke",
                run_id="smoke-seed-7-test",
                source_sha256=runner.MINI_RADAS_SOURCE_SHA256,
                paths={"trajectories_csv": output / "trajectory.csv"},
            )
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["orchestrator"]["expected_commit"],
                runner.MINI_RADAS_COMMIT,
            )
            self.assertEqual(payload["completed_trials"], 1)
            self.assertEqual(payload["expected_trials"], 1)
            self.assertEqual(
                payload["fixed_parameters"],
                {
                    "initial_backward_weight": 0.5,
                    "theta": 0.4,
                    "learning_rate": 0.005,
                    "backward_learning_rate": 0.005,
                    "dt": 0.1,
                    "tmax": 50.0,
                    "delay": 5.0,
                    "output_delay_std": 0.0,
                },
            )

    def test_run_uses_public_local_api(self) -> None:
        calls: dict[str, object] = {}

        async def fake_run_experiment(**kwargs):
            calls.update(kwargs)
            return {"df": pd.DataFrame([{"complete": True}])}

        fake_radas = types.ModuleType("radas")
        fake_radas.run_experiment = fake_run_experiment
        spec = runner.ExperimentSpec(
            trainable=runner.experiment.spiking_trajectory_trainable,
            param_space={},
            tune_config=object(),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "result.csv"
            artifact.write_text("value\n1\n", encoding="utf-8")
            args = runner.build_parser().parse_args(
                [
                    "--smoke",
                    "--storage",
                    str(root / "storage"),
                    "--output",
                    str(root / "output"),
                    "--no-plots",
                ]
            )
            with (
                mock.patch.dict(sys.modules, {"radas": fake_radas}),
                mock.patch.object(
                    runner, "validate_mini_radas", return_value="verified-sha"
                ),
                mock.patch.object(
                    runner, "build_experiment_spec", return_value=spec
                ),
                mock.patch.object(
                    runner,
                    "materialize_results",
                    return_value={"result": artifact},
                ),
            ):
                paths = asyncio.run(runner.run(args))
            payload = json.loads(paths["run_metadata"].read_text(encoding="utf-8"))

        self.assertEqual(calls["run_with"], "local")
        self.assertEqual(calls["dos"], ["run", "analyze"])
        self.assertIs(
            calls["trainable"], runner.experiment.spiking_trajectory_trainable
        )
        self.assertEqual(
            payload["orchestrator"]["verified_source_sha256"], "verified-sha"
        )


if __name__ == "__main__":
    unittest.main()
