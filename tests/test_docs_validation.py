from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


class TestDocsValidation(unittest.TestCase):
    def test_docs_validation_script_passes(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "scripts/validate_docs.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            self.fail(
                "docs validation should pass in repository baseline.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        self.assertIn("Docs validation passed.", result.stdout)


if __name__ == "__main__":
    unittest.main()
