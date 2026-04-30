"""Compatibility tests for the Reward Harness renaming.

Ensures that ``reward_harness.*`` and ``autosr.*`` resolve to the same
objects and that both CLI families exit successfully.
"""

from __future__ import annotations

import subprocess
import sys
import unittest

import autosr.data_models
import autosr.harness
import autosr.rm
import autosr.rl
import autosr.search
import autosr.types
import reward_harness
import reward_harness.data_models
import reward_harness.harness
import reward_harness.rm
import reward_harness.rl
import reward_harness.search
import reward_harness.types


class TestRewardHarnessImportCompat(unittest.TestCase):
    def test_top_level_rubric_identity(self) -> None:
        self.assertIs(reward_harness.Rubric, autosr.Rubric)

    def test_top_level_search_result_identity(self) -> None:
        self.assertIs(reward_harness.SearchResult, autosr.SearchResult)

    def test_data_models_rubric_identity(self) -> None:
        self.assertIs(
            reward_harness.data_models.Rubric,
            autosr.data_models.Rubric,
        )

    def test_types_backend_type_identity(self) -> None:
        self.assertIs(
            reward_harness.types.BackendType,
            autosr.types.BackendType,
        )

    def test_search_evolutionary_config_identity(self) -> None:
        self.assertIs(
            reward_harness.search.EvolutionaryConfig,
            autosr.search.EvolutionaryConfig,
        )

    def test_harness_search_session_identity(self) -> None:
        self.assertIs(
            reward_harness.harness.SearchSession,
            autosr.harness.SearchSession,
        )

    def test_rm_artifact_identity(self) -> None:
        self.assertIs(
            reward_harness.rm.RMArtifact,
            autosr.rm.RMArtifact,
        )

    def test_rl_training_manifest_identity(self) -> None:
        self.assertIs(
            reward_harness.rl.TrainingManifest,
            autosr.rl.TrainingManifest,
        )


class TestRewardHarnessCliCompat(unittest.TestCase):
    def _run_help(self, module: str) -> int:
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            capture_output=True,
            text=True,
        )
        return result.returncode

    def test_new_cli_help_exit_code(self) -> None:
        self.assertEqual(self._run_help("reward_harness.cli"), 0)

    def test_old_cli_help_exit_code(self) -> None:
        self.assertEqual(self._run_help("autosr.cli"), 0)

    def test_rm_export_help_exit_code(self) -> None:
        self.assertEqual(self._run_help("reward_harness.rm.export"), 0)
        self.assertEqual(self._run_help("autosr.rm.export"), 0)

    def test_rm_server_help_exit_code(self) -> None:
        self.assertEqual(self._run_help("reward_harness.rm.server"), 0)
        self.assertEqual(self._run_help("autosr.rm.server"), 0)

    def test_rl_record_manifest_help_exit_code(self) -> None:
        self.assertEqual(self._run_help("reward_harness.rl.record_manifest"), 0)
        self.assertEqual(self._run_help("autosr.rl.cli.record_manifest"), 0)

    def test_rl_show_lineage_help_exit_code(self) -> None:
        self.assertEqual(self._run_help("reward_harness.rl.show_lineage"), 0)
        self.assertEqual(self._run_help("autosr.rl.cli.show_lineage"), 0)

    def test_rl_compare_runs_help_exit_code(self) -> None:
        self.assertEqual(self._run_help("reward_harness.rl.compare_runs"), 0)
        self.assertEqual(self._run_help("autosr.rl.cli.compare_runs"), 0)

    def test_rl_check_regression_help_exit_code(self) -> None:
        self.assertEqual(self._run_help("reward_harness.rl.check_regression"), 0)
        self.assertEqual(self._run_help("autosr.rl.cli.check_regression"), 0)


if __name__ == "__main__":
    unittest.main()
