from __future__ import annotations

import unittest

from autosr.config import LLMBackendConfig, RuntimeConfig
from autosr.cli import build_parser
from autosr.types import BackendType


class TestCliBackendSelection(unittest.TestCase):
    def test_default_model_is_stepfun_flash(self) -> None:
        self.assertEqual(LLMBackendConfig().default_model, "stepfun/step-3.5-flash:free")

        parser = build_parser()
        args = parser.parse_args(["--dataset", "d.json", "--output", "o.json"])
        self.assertEqual(args.model_default, "stepfun/step-3.5-flash:free")

    def test_auto_uses_mock_without_key(self) -> None:
        config = RuntimeConfig(backend="auto", llm=LLMBackendConfig(api_key=None))
        self.assertEqual(config.resolve_backend(), BackendType.MOCK)

    def test_auto_uses_llm_with_key(self) -> None:
        config = RuntimeConfig(backend="auto", llm=LLMBackendConfig(api_key="sk-test"))
        self.assertEqual(config.resolve_backend(), BackendType.LLM)

    def test_explicit_mock_ignores_key(self) -> None:
        config = RuntimeConfig(backend="mock", llm=LLMBackendConfig(api_key="sk-test"))
        self.assertEqual(config.resolve_backend(), BackendType.MOCK)

    def test_llm_without_key_raises(self) -> None:
        config = RuntimeConfig(backend="llm", llm=LLMBackendConfig(api_key=None))
        with self.assertRaises(ValueError):
            config.resolve_backend()

    def test_role_model_fallbacks(self) -> None:
        llm_config = LLMBackendConfig(
            default_model="openai/gpt-4o-mini",
            initializer_model=None,
            proposer_model="anthropic/claude-3.5-sonnet",
            verifier_model=None,
            judge_model=None,
        )
        self.assertEqual(llm_config.get_model_for_role("initializer"), "openai/gpt-4o-mini")
        self.assertEqual(llm_config.get_model_for_role("proposer"), "anthropic/claude-3.5-sonnet")
        self.assertEqual(llm_config.get_model_for_role("verifier"), "openai/gpt-4o-mini")
        self.assertEqual(llm_config.get_model_for_role("judge"), "openai/gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
