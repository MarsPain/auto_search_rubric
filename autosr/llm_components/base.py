from __future__ import annotations

import json
import logging
from typing import Any, Callable, Protocol

from ..exceptions import LLMParseError
from ..prompts.loader import ConstantPromptRepository, PromptRepository

logger = logging.getLogger("autosr.llm_components")

_REQUIRED_CRITERION_FIELDS = ["criterion_id", "text", "weight"]


def get_required_fields_json() -> str:
    """Return JSON-encoded required criterion fields."""
    return json.dumps(_REQUIRED_CRITERION_FIELDS)


class JsonRequester(Protocol):
    def request_json(self, *, model: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Call an LLM and return a parsed JSON object."""


class LLMComponentBase:
    def __init__(
        self,
        requester: JsonRequester,
        *,
        model: str,
        max_retries: int,
        prompt_repository: PromptRepository | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.requester = requester
        self.model = model
        self.max_retries = max_retries
        self._repository = prompt_repository or ConstantPromptRepository()

    def _request_validated(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        parser: Callable[[dict[str, Any]], Any],
    ) -> Any:
        attempts = self.max_retries + 1
        last_error: LLMParseError | None = None
        for attempt in range(attempts):
            payload = self.requester.request_json(
                model=self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            try:
                return parser(payload)
            except LLMParseError as exc:
                last_error = exc
                if attempt == attempts - 1:
                    raise
        raise LLMParseError(f"failed to parse payload after retries: {last_error}")

    def _get_prompt_config(self, template_id: str, version: str | None = None):
        return self._repository.get(template_id, version)

    def _render_with_fallback(
        self,
        template_id: str,
        context: dict[str, Any],
        fallback_system: str,
        fallback_template: str,
    ) -> tuple[str, str]:
        try:
            config = self._get_prompt_config(template_id)
            return config.render(**context)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            logger.debug("Using constant fallback for %s: %s", template_id, exc)
            return (
                fallback_system,
                fallback_template.format(**context),
            )
