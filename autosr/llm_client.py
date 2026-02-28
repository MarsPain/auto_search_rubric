from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any

from .exceptions import LLMCallError, LLMParseError
from .llm_config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChatRequest:
    model: str
    system_prompt: str
    user_prompt: str


class LLMClient:
    def __init__(self, config: LLMConfig, *, client: Any | None = None) -> None:
        self.config = config
        self._client = client or _create_openai_client(config)

    def request_json(self, *, model: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        request = ChatRequest(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
        last_error: Exception | None = None
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            try:
                response_text = self._request_text(request)
                result = extract_json_object(response_text)
                logger.debug("LLM request_json response: %s", json.dumps(result, ensure_ascii=False, indent=2))
                return result
            except (LLMCallError, LLMParseError) as exc:
                last_error = exc
                logger.warning("LLM request_json attempt %d/%d failed: %s", attempt + 1, attempts, exc)
                if attempt == attempts - 1:
                    raise
        raise LLMCallError(f"exhausted retries for model={model}: {last_error}")

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
            raise LLMCallError(f"failed to call LLM model={request.model}: {exc}") from exc

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
