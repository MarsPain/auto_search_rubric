from __future__ import annotations

from pathlib import Path
import re
import unittest


class TestNoHardcodedSecrets(unittest.TestCase):
    def test_llm_integration_file_has_no_hardcoded_secret_or_shell_exports(
        self,
    ) -> None:
        target = Path(__file__).resolve().parent / "test_llm_integration.py"
        content = target.read_text(encoding="utf-8")

        banned_snippets = (
            "export LLM_API_KEY=",
            "export ANTHROPIC_API_KEY=",
            "CODEx prompt style",
        )
        for snippet in banned_snippets:
            self.assertNotIn(snippet, content)

        # Catch common key-like token patterns (e.g. sk-...).
        self.assertIsNone(re.search(r"\bsk-[A-Za-z0-9_-]{20,}\b", content))


if __name__ == "__main__":
    unittest.main()
