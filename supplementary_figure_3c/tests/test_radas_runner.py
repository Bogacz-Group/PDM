from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

SPEC = importlib.util.spec_from_file_location(
    "supplementary_figure_3c_radas_runner", MODULE_DIR / "run_with_radas.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

import weight_trajectories as trajectories  # noqa: E402


def profile_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "quick": False,
        "smoke": False,
        "iterations": None,
        "grid_min": None,
        "grid_max": None,
        "grid_step": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class RadasRunnerTests(unittest.TestCase):
    def test_profiles_are_bounded_and_reproducible(self) -> None:
        full = runner.resolve_profile(profile_args())
        quick = runner.resolve_profile(profile_args(quick=True))
        smoke = runner.resolve_profile(profile_args(smoke=True))
        self.assertEqual((full.iterations, len(runner.profile_grid(full))), (128, 9))
        self.assertEqual((quick.iterations, len(runner.profile_grid(quick))), (8, 3))
        self.assertEqual((smoke.iterations, runner.profile_grid(smoke)), (1, [0.0]))

    def test_storage_components_reject_traversal(self) -> None:
        self.assertEqual(runner.validate_storage_component("safe-name", "test"), "safe-name")
        for unsafe in ("../escape", "a/b", ".", "two words"):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                runner.validate_storage_component(unsafe, "test")

    def test_source_validation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            for filename in runner.MINI_RADAS_SOURCE_FILES:
                (package / filename).write_text("# not pinned\n", encoding="utf-8")

            class Module:
                __file__ = str(package / "__init__.py")

            def run_experiment(
                user_name=None,
                resources=None,
                run_with=None,
                local_storage_path=None,
                dos=None,
            ):
                return None

            with self.assertRaisesRegex(RuntimeError, "do not match"):
                runner.validate_mini_radas(Module, run_experiment)

    def test_materialize_writes_one_trial(self) -> None:
        result = trajectories.run_trajectory(-0.5, 1.0, num_dataset_iterations=2)
        frame = pd.DataFrame([result])
        with tempfile.TemporaryDirectory() as directory:
            paths = runner.materialize(frame, Path(directory), plots=False)
            rows = pd.read_csv(paths["trajectories_csv"])
            trials = pd.read_csv(paths["trials_csv"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(trials), 1)
        self.assertEqual(int(rows.iloc[-1]["sample_updates"]), 8)

    def test_run_uses_public_local_api(self) -> None:
        calls: dict[str, object] = {}

        async def fake_run_experiment(**kwargs):
            calls.update(kwargs)
            return {"df": pd.DataFrame([{"complete": True}])}

        fake_radas = types.ModuleType("radas")
        fake_radas.run_experiment = fake_run_experiment
        fake_ray = types.ModuleType("ray")
        fake_ray.tune = types.SimpleNamespace(
            grid_search=lambda values: values,
            TuneConfig=lambda: object(),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
                mock.patch.dict(sys.modules, {"radas": fake_radas, "ray": fake_ray}),
                mock.patch.object(
                    runner, "validate_mini_radas", return_value="verified-sha"
                ),
                mock.patch.object(
                    runner, "materialize", return_value={"result": artifact}
                ),
            ):
                paths = asyncio.run(runner.run(args))
            metadata = json.loads(
                paths["run_metadata"].read_text(encoding="utf-8")
            )

        self.assertEqual(calls["run_with"], "local")
        self.assertEqual(calls["dos"], ["run", "analyze"])
        self.assertIs(calls["trainable"], trajectories.firing_rate_trajectory_trainable)
        self.assertEqual(
            metadata["orchestrator"]["verified_source_sha256"], "verified-sha"
        )
        self.assertEqual(metadata["completed_trials"], 1)


if __name__ == "__main__":
    unittest.main()
