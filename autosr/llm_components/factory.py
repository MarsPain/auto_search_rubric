from __future__ import annotations

from ..prompts.loader import create_repository
from .base import JsonRequester
from .use_cases import (
    LLMPreferenceJudge,
    LLMRubricInitializer,
    LLMRubricProposer,
    LLMVerifier,
)


def create_llm_components(
    requester: JsonRequester,
    *,
    model: str,
    max_retries: int = 3,
    prompt_config_path: str | None = None,
):
    """Factory to create all LLM components with optional external configuration."""
    repository = create_repository(prompt_config_path) if prompt_config_path else None

    initializer = LLMRubricInitializer(
        requester,
        model=model,
        max_retries=max_retries,
        prompt_repository=repository,
    )
    proposer = LLMRubricProposer(
        requester,
        model=model,
        max_retries=max_retries,
        prompt_repository=repository,
    )
    verifier = LLMVerifier(
        requester,
        model=model,
        max_retries=max_retries,
        prompt_repository=repository,
    )
    judge = LLMPreferenceJudge(
        requester,
        model=model,
        max_retries=max_retries,
        prompt_repository=repository,
    )

    return initializer, proposer, verifier, judge
