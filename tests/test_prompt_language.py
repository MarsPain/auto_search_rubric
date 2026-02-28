from __future__ import annotations

import unittest

from autosr.cli import build_parser, build_runtime_config
from autosr.config import LLMBackendConfig, RuntimeConfig
from autosr.factory import ComponentFactory
from autosr.prompts.loader import PromptConfig


class _DummyRequester:
    def request_json(self, *, model: str, system_prompt: str, user_prompt: str):  # noqa: ANN001
        raise AssertionError("Dummy requester should not be called in this test")


class _InjectedPromptRepository:
    def get(self, template_id: str, version: str | None = None) -> PromptConfig:
        return PromptConfig(
            template_id=template_id,
            version=version or "injected",
            system="Injected system prompt",
            user_template="{prompt_json}",
            variables=["prompt_json"],
        )

    def list_versions(self, template_id: str) -> list[str]:
        return ["injected"]


class TestPromptLanguage(unittest.TestCase):
    def test_cli_parses_prompt_language(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "d.json",
                "--output",
                "o.json",
                "--prompt-language",
                "zh",
            ]
        )
        self.assertEqual(args.prompt_language, "zh")

    def test_runtime_config_wires_prompt_language(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "d.json",
                "--output",
                "o.json",
                "--backend",
                "llm",
                "--prompt-language",
                "zh",
            ]
        )
        config = build_runtime_config(args)
        self.assertEqual(config.llm.prompt_language, "zh")

    def test_factory_uses_zh_prompt_templates_when_configured(self) -> None:
        requester = _DummyRequester()
        config = RuntimeConfig(
            backend="llm",
            llm=LLMBackendConfig(
                api_key="sk-test",
                default_model="unit-test-model",
                prompt_language="zh",
            ),
        )
        factory = ComponentFactory(config, llm_requester=requester)

        initializer = factory.create_base_initializer()
        self.assertIs(initializer.requester, requester)  # type: ignore[attr-defined]
        prompt_config = initializer._get_prompt_config("rubric_initializer")  # type: ignore[attr-defined]
        self.assertIn("你是 rubric 设计师", prompt_config.system)

    def test_factory_falls_back_to_default_prompts_dir(self) -> None:
        requester = _DummyRequester()
        config = RuntimeConfig(
            backend="llm",
            llm=LLMBackendConfig(
                api_key="sk-test",
                default_model="unit-test-model",
                prompt_language="xx",
            ),
        )
        factory = ComponentFactory(config, llm_requester=requester)

        initializer = factory.create_base_initializer()
        self.assertIs(initializer.requester, requester)  # type: ignore[attr-defined]
        prompt_config = initializer._get_prompt_config("rubric_initializer")  # type: ignore[attr-defined]
        self.assertIn("You are a rubric designer", prompt_config.system)

    def test_factory_uses_injected_prompt_repository(self) -> None:
        config = RuntimeConfig(
            backend="llm",
            llm=LLMBackendConfig(
                api_key="sk-test",
                default_model="unit-test-model",
                prompt_language="zh",
            ),
        )
        repository = _InjectedPromptRepository()
        factory = ComponentFactory(
            config,
            llm_requester=_DummyRequester(),
            prompt_repository=repository,
        )

        initializer = factory.create_base_initializer()
        prompt_config = initializer._get_prompt_config("rubric_initializer")  # type: ignore[attr-defined]
        self.assertEqual(prompt_config.system, "Injected system prompt")


if __name__ == "__main__":
    unittest.main()
