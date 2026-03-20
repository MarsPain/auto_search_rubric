from __future__ import annotations

from types import SimpleNamespace
import unittest

from autosr.exceptions import LLMCallError, LLMFatalCallError
from autosr.llm_client import LLMClient
from autosr.llm_config import LLMConfig


def _response_with_text(text: str) -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class _RetryingChatCompletions:
    def __init__(self, failures_before_success: int) -> None:
        self.failures_before_success = failures_before_success
        self.calls = 0

    def create(self, **kwargs):  # noqa: ANN003, ANN201
        del kwargs
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise TimeoutError("request timed out")
        return _response_with_text('{"ok": true}')


class _AlwaysFailingChatCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):  # noqa: ANN003, ANN201
        del kwargs
        self.calls += 1
        raise TimeoutError("request timed out")


class _FatalAuthError(Exception):
    def __init__(self) -> None:
        super().__init__("Error code: 401 - {'error': {'message': 'User not found.', 'code': 401}}")
        self.status_code = 401


class _AlwaysAuthFailingChatCompletions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):  # noqa: ANN003, ANN201
        del kwargs
        self.calls += 1
        raise _FatalAuthError()


class TestLLMClientResilience(unittest.TestCase):
    def test_request_json_retries_with_exponential_backoff(self) -> None:
        sleeps: list[float] = []
        completions = _RetryingChatCompletions(failures_before_success=2)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        config = LLMConfig(
            base_url="https://example.com",
            api_key="sk-test",
            timeout=30.0,
            max_retries=3,
            retry_backoff_base=1.0,
            retry_backoff_max=10.0,
            retry_jitter=0.0,
        )
        llm = LLMClient(config, client=client, sleep=sleeps.append)

        payload = llm.request_json(
            model="openai/gpt-4o-mini",
            system_prompt="system",
            user_prompt="user",
        )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(sleeps, [1.0, 2.0])
        self.assertEqual(completions.calls, 3)

    def test_request_json_applies_jitter_within_bounds(self) -> None:
        sleeps: list[float] = []
        completions = _RetryingChatCompletions(failures_before_success=1)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        config = LLMConfig(
            base_url="https://example.com",
            api_key="sk-test",
            timeout=30.0,
            max_retries=2,
            retry_backoff_base=2.0,
            retry_backoff_max=10.0,
            retry_jitter=0.5,
        )
        llm = LLMClient(config, client=client, sleep=sleeps.append, random_seed=7)

        llm.request_json(
            model="openai/gpt-4o-mini",
            system_prompt="system",
            user_prompt="user",
        )

        self.assertEqual(len(sleeps), 1)
        self.assertGreaterEqual(sleeps[0], 1.0)
        self.assertLessEqual(sleeps[0], 3.0)
        self.assertNotEqual(sleeps[0], 2.0)

    def test_request_json_raises_after_exhausted_retries(self) -> None:
        sleeps: list[float] = []
        completions = _AlwaysFailingChatCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        config = LLMConfig(
            base_url="https://example.com",
            api_key="sk-test",
            timeout=30.0,
            max_retries=2,
            retry_backoff_base=1.0,
            retry_backoff_max=10.0,
            retry_jitter=0.0,
        )
        llm = LLMClient(config, client=client, sleep=sleeps.append)

        with self.assertRaises(LLMCallError):
            llm.request_json(
                model="openai/gpt-4o-mini",
                system_prompt="system",
                user_prompt="user",
            )

        self.assertEqual(sleeps, [1.0, 2.0])
        self.assertEqual(completions.calls, 3)

    def test_request_json_does_not_retry_fatal_auth_error(self) -> None:
        sleeps: list[float] = []
        completions = _AlwaysAuthFailingChatCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        config = LLMConfig(
            base_url="https://example.com",
            api_key="sk-test",
            timeout=30.0,
            max_retries=3,
            retry_backoff_base=1.0,
            retry_backoff_max=10.0,
            retry_jitter=0.0,
        )
        llm = LLMClient(config, client=client, sleep=sleeps.append)

        with self.assertRaises(LLMFatalCallError):
            llm.request_json(
                model="openai/gpt-4o-mini",
                system_prompt="system",
                user_prompt="user",
            )

        self.assertEqual(sleeps, [])
        self.assertEqual(completions.calls, 1)


if __name__ == "__main__":
    unittest.main()
