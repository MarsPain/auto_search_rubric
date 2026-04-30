from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import random
import time
from typing import Any, Callable

from .exceptions import LLMCallError, LLMFatalCallError, LLMParseError
from .llm_config import LLMConfig

logger = logging.getLogger(__name__)
_FATAL_STATUS_CODES = {400, 401, 403, 404}
_FATAL_MESSAGE_HINTS = (
    "user not found",
    "invalid api key",
    "unauthorized",
    "authentication",
    "forbidden",
    "permission denied",
    "no such model",
    "model not found",
)


@dataclass(slots=True)
class ChatRequest:
    model: str
    system_prompt: str
    user_prompt: str


class LLMClient:
    def __init__(
        self,
        config: LLMConfig,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.config = config
        self._client = client or _create_openai_client(config)
        self._sleep = sleep or time.sleep
        self._rng = random.Random(random_seed)

    def request_json(
        self, *, model: str, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        request = ChatRequest(
            model=model, system_prompt=system_prompt, user_prompt=user_prompt
        )
        last_error: Exception | None = None
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            try:
                response_text = self._request_text(request)
                result = extract_json_object(response_text)
                logger.debug(
                    "LLM request_json response: %s",
                    json.dumps(result, ensure_ascii=False, indent=2),
                )
                return result
            except (LLMCallError, LLMParseError) as exc:
                last_error = exc
                logger.warning(
                    "LLM request_json attempt %d/%d failed: %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
                if isinstance(exc, LLMFatalCallError):
                    raise
                if attempt == attempts - 1:
                    raise
                delay_seconds = self._compute_retry_delay(attempt)
                if delay_seconds > 0:
                    logger.warning(
                        "LLM request_json backing off for %.2fs before retry (attempt %d/%d)",
                        delay_seconds,
                        attempt + 2,
                        attempts,
                    )
                    self._sleep(delay_seconds)
        raise LLMCallError(f"exhausted retries for model={model}: {last_error}")

    def _compute_retry_delay(self, attempt_index: int) -> float:
        base_delay = self.config.retry_backoff_base * (2**attempt_index)
        bounded_delay = min(base_delay, self.config.retry_backoff_max)
        if bounded_delay <= 0:
            return 0.0

        if self.config.retry_jitter <= 0:
            return bounded_delay

        jitter_factor = self._rng.uniform(
            1.0 - self.config.retry_jitter,
            1.0 + self.config.retry_jitter,
        )
        jittered_delay = bounded_delay * jitter_factor
        if jittered_delay <= 0:
            return 0.0
        return min(jittered_delay, self.config.retry_backoff_max)

    def _request_text(self, request: ChatRequest) -> str:
        try:
            response = self._client.chat.completions.create(
                model=request.model,
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
                temperature=self.config.temperature,
                timeout=self.config.timeout,
            )
        except Exception as exc:  # pragma: no cover - external SDK behavior
            error_cls = (
                LLMFatalCallError if _is_fatal_call_exception(exc) else LLMCallError
            )
            raise error_cls(f"failed to call LLM model={request.model}: {exc}") from exc

        text = _extract_content_text(response)
        logger.debug("LLM raw response text: %s", text)
        if not text.strip():
            raise LLMParseError(f"empty response from model={request.model}")
        return text


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    payload = _loads_json_object(text)
    if payload is not None:
        return payload

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx < 0 or end_idx <= start_idx:
        raise LLMParseError("could not locate a JSON object in response")

    sliced = text[start_idx : end_idx + 1]
    payload = _loads_json_object(sliced)
    if payload is None:
        raise LLMParseError("failed to decode JSON object from response")
    return payload


def _loads_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _extract_content_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise LLMParseError("response contains no choices")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise LLMParseError("response choice contains no message")

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks)

    logger.debug("content %s", str(content))

    return str(content)


def _create_openai_client(config: LLMConfig) -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - dependency availability varies
        raise ImportError(
            "openai package is required for LLM backend. Install dependencies first."
        ) from exc
    return OpenAI(base_url=config.base_url, api_key=config.api_key)


def _is_fatal_call_exception(exc: Exception) -> bool:
    status_code = _extract_status_code(exc)
    if status_code in _FATAL_STATUS_CODES:
        return True

    lowered = str(exc).lower()
    return any(hint in lowered for hint in _FATAL_MESSAGE_HINTS)


def _extract_status_code(exc: Exception) -> int | None:
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int):
        return direct

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status

    return None
