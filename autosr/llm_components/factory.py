from __future__ import annotations

import warnings

from ..llm_config import RoleModelConfig
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
) -> tuple[LLMRubricInitializer, LLMRubricProposer, LLMVerifier, LLMPreferenceJudge]:
    """Factory to create LLM components.

    Note:
        This helper is kept for backward compatibility. New runtime code should
        use ``autosr.factory.ComponentFactory`` for end-to-end wiring.
    """
    warnings.warn(
        "create_llm_components is a legacy helper; prefer ComponentFactory.",
        DeprecationWarning,
        stacklevel=2,
    )
    repository = create_repository(prompt_config_path) if prompt_config_path else None
    models = RoleModelConfig(default=model)

    initializer = LLMRubricInitializer(
        requester,
        model=models.for_role("initializer"),
        max_retries=max_retries,
        prompt_repository=repository,
    )
    proposer = LLMRubricProposer(
        requester,
        model=models.for_role("proposer"),
        max_retries=max_retries,
        prompt_repository=repository,
    )
    verifier = LLMVerifier(
        requester,
        model=models.for_role("verifier"),
        max_retries=max_retries,
        prompt_repository=repository,
    )
    judge = LLMPreferenceJudge(
        requester,
        model=models.for_role("judge"),
        max_retries=max_retries,
        prompt_repository=repository,
    )

    return initializer, proposer, verifier, judge
