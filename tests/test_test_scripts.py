from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


class TestTestScripts(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.scripts_dir = self.repo_root / "scripts"

    def test_unit_and_integration_test_scripts_exist(self) -> None:
        self.assertTrue((self.scripts_dir / "run_tests_unit.sh").exists())
        self.assertTrue((self.scripts_dir / "run_tests_integration.sh").exists())

    def test_run_tests_aggregates_unit_and_integration(self) -> None:
        run_tests = (self.scripts_dir / "run_tests.sh").read_text(encoding="utf-8")
        self.assertIn("run_tests_unit.sh", run_tests)
        self.assertIn("run_tests_integration.sh", run_tests)

    def test_integration_script_fails_without_api_key(self) -> None:
        integration_script = self.scripts_dir / "run_tests_integration.sh"
        result = subprocess.run(
            ["bash", str(integration_script)],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            env={},
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LLM_API_KEY is not set", result.stderr)

    def test_test_scripts_support_virtualenv_python(self) -> None:
        unit_script = (self.scripts_dir / "run_tests_unit.sh").read_text(encoding="utf-8")
        integration_script = (self.scripts_dir / "run_tests_integration.sh").read_text(
            encoding="utf-8"
        )
        formal_script = (self.scripts_dir / "run_formal_search.sh").read_text(
            encoding="utf-8"
        )
        for script_content in (unit_script, integration_script, formal_script):
            self.assertIn("command -v uv", script_content)
            self.assertIn("uv run python", script_content)
            self.assertIn("VIRTUAL_ENV", script_content)
            self.assertIn("python3", script_content)


if __name__ == "__main__":
    unittest.main()
