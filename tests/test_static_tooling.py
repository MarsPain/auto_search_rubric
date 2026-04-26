from __future__ import annotations

from pathlib import Path
import tomllib
import unittest


class TestStaticTooling(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent

    def test_ruff_configuration_is_declared(self) -> None:
        pyproject = tomllib.loads(
            (self.repo_root / "pyproject.toml").read_text(encoding="utf-8")
        )

        ruff_config = pyproject.get("tool", {}).get("ruff", {})
        self.assertEqual(ruff_config.get("target-version"), "py311")
        self.assertIn("autosr", ruff_config.get("src", []))
        self.assertIn("tests", ruff_config.get("src", []))
        self.assertGreaterEqual(ruff_config.get("line-length", 0), 88)

        lint_config = ruff_config.get("lint", {})
        selected_rules = set(lint_config.get("select", []))
        self.assertGreaterEqual({"E", "F", "I", "UP", "B"}, selected_rules)

    def test_quality_check_script_runs_ruff_checks(self) -> None:
        script_path = self.repo_root / "scripts" / "run_quality_checks.sh"
        self.assertTrue(script_path.exists())

        script_content = script_path.read_text(encoding="utf-8")
        self.assertIn("command -v uv", script_content)
        self.assertIn("RUFF_CMD=(uv run --with ruff ruff)", script_content)
        self.assertIn('"${RUFF_CMD[@]}" check autosr tests scripts', script_content)
        self.assertIn(
            '"${RUFF_CMD[@]}" format --check autosr tests scripts',
            script_content,
        )
        self.assertIn("MYPY_CMD=(uv run --with mypy mypy)", script_content)
        self.assertIn('"${MYPY_CMD[@]}" autosr/mix_reward.py', script_content)

    def test_mypy_configuration_is_declared(self) -> None:
        pyproject = tomllib.loads(
            (self.repo_root / "pyproject.toml").read_text(encoding="utf-8")
        )

        mypy_config = pyproject.get("tool", {}).get("mypy", {})
        self.assertEqual("3.11", mypy_config.get("python_version"))
        self.assertTrue(mypy_config.get("strict"))
        self.assertEqual("skip", mypy_config.get("follow_imports"))


if __name__ == "__main__":
    unittest.main()
