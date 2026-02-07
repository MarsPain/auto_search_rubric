from __future__ import annotations

import argparse
import unittest

from autosr.cli import build_role_model_config, resolve_backend


class TestCliBackendSelection(unittest.TestCase):
    def test_auto_uses_mock_without_key(self) -> None:
        self.assertEqual(resolve_backend("auto", None), "mock")

    def test_auto_uses_llm_with_key(self) -> None:
        self.assertEqual(resolve_backend("auto", "sk-test"), "llm")

    def test_explicit_mock_ignores_key(self) -> None:
        self.assertEqual(resolve_backend("mock", "sk-test"), "mock")

    def test_llm_without_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_backend("llm", None)

    def test_role_model_fallbacks(self) -> None:
        args = argparse.Namespace(
            model_default="openai/gpt-4o-mini",
            model_initializer=None,
            model_proposer="anthropic/claude-3.5-sonnet",
            model_verifier=None,
            model_judge=None,
        )
        model_cfg = build_role_model_config(args)
        self.assertEqual(model_cfg.for_role("initializer"), "openai/gpt-4o-mini")
        self.assertEqual(model_cfg.for_role("proposer"), "anthropic/claude-3.5-sonnet")
        self.assertEqual(model_cfg.for_role("verifier"), "openai/gpt-4o-mini")
        self.assertEqual(model_cfg.for_role("judge"), "openai/gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
