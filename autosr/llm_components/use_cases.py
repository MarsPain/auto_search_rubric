from __future__ import annotations

import json
import random
from typing import Any

from ..llm_client import LLMParseError
from ..models import GradingProtocol, PromptExample, ResponseCandidate, Rubric
from ..prompts import constants as prompt_constants
from ..prompts.loader import PromptRepository
from ..types import MutationMode
from .base import LLMComponentBase, JsonRequester, get_required_fields_json
from .parsers import (
    candidate_snapshot,
    normalize_grade,
    normalize_preference,
    payload_to_rubric,
)


class LLMRubricInitializer(LLMComponentBase):
    """Initializes rubrics using LLM."""

    def __init__(
        self,
        requester: JsonRequester,
        *,
        model: str,
        max_retries: int,
        prompt_repository: PromptRepository | None = None,
    ) -> None:
        super().__init__(
            requester,
            model=model,
            max_retries=max_retries,
            prompt_repository=prompt_repository,
        )

    def initialize(self, item: PromptExample, *, rng: random.Random) -> Rubric:
        candidate_samples = [candidate_snapshot(c) for c in item.candidates[:4]]
        context = {
            "prompt_json": json.dumps(item.prompt, ensure_ascii=False),
            "candidate_samples_json": json.dumps(candidate_samples, ensure_ascii=False),
            "required_fields_json": get_required_fields_json(),
        }

        system_prompt, user_prompt = self._render_with_fallback(
            template_id="rubric_initializer",
            context=context,
            fallback_system=prompt_constants.RUBRIC_INITIALIZER_SYSTEM,
            fallback_template=prompt_constants.RUBRIC_INITIALIZER_USER_TEMPLATE,
        )

        fallback_id = f"{item.prompt_id}_llm_init_{rng.randint(0, 9999)}"
        return self._request_validated(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parser=lambda payload: payload_to_rubric(
                payload,
                fallback_rubric_id=fallback_id,
                fallback_protocol=GradingProtocol(num_votes=1, allow_na=True, vote_method="majority"),
            ),
        )


class LLMRubricProposer(LLMComponentBase):
    """Proposes rubric mutations using LLM."""

    def __init__(
        self,
        requester: JsonRequester,
        *,
        model: str,
        max_retries: int,
        prompt_repository: PromptRepository | None = None,
    ) -> None:
        super().__init__(
            requester,
            model=model,
            max_retries=max_retries,
            prompt_repository=prompt_repository,
        )

    def propose(
        self,
        prompt: str,
        left: ResponseCandidate,
        right: ResponseCandidate,
        rubric: Rubric,
        *,
        mode: MutationMode,
        rng: random.Random,
    ) -> Rubric:
        context = {
            "mode": mode.value,
            "prompt_json": json.dumps(prompt, ensure_ascii=False),
            "current_rubric_json": json.dumps(rubric.to_dict(), ensure_ascii=False),
            "top_candidate_json": json.dumps(candidate_snapshot(left), ensure_ascii=False),
            "runner_up_candidate_json": json.dumps(candidate_snapshot(right), ensure_ascii=False),
            "required_fields_json": get_required_fields_json(),
        }

        system_prompt, user_prompt = self._render_with_fallback(
            template_id="rubric_proposer",
            context=context,
            fallback_system=prompt_constants.RUBRIC_PROPOSER_SYSTEM,
            fallback_template=prompt_constants.RUBRIC_PROPOSER_USER_TEMPLATE,
        )

        fallback_id = f"{rubric.rubric_id}_{mode.value}_{rng.randint(0, 9999)}"
        return self._request_validated(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parser=lambda payload: payload_to_rubric(
                payload,
                fallback_rubric_id=fallback_id,
                fallback_protocol=rubric.grading_protocol,
            ),
        )


class LLMVerifier(LLMComponentBase):
    """Verifies candidate responses using LLM."""

    def __init__(
        self,
        requester: JsonRequester,
        *,
        model: str,
        max_retries: int,
        prompt_repository: PromptRepository | None = None,
    ) -> None:
        super().__init__(
            requester,
            model=model,
            max_retries=max_retries,
            prompt_repository=prompt_repository,
        )

    def grade(
        self,
        prompt: str,
        candidate: ResponseCandidate,
        rubric: Rubric,
        *,
        seed: int,
    ) -> dict[str, int | None]:
        criteria_data = [criterion.to_dict() for criterion in rubric.criteria]
        allow_na = "true" if rubric.grading_protocol.allow_na else "false"

        context = {
            "seed": seed,
            "prompt_json": json.dumps(prompt, ensure_ascii=False),
            "candidate_json": json.dumps(candidate_snapshot(candidate), ensure_ascii=False),
            "criteria_json": json.dumps(criteria_data, ensure_ascii=False),
            "allow_na": allow_na,
        }

        system_prompt, user_prompt = self._render_with_fallback(
            template_id="verifier",
            context=context,
            fallback_system=prompt_constants.VERIFIER_SYSTEM,
            fallback_template=prompt_constants.VERIFIER_USER_TEMPLATE,
        )

        def parse(payload: dict[str, Any]) -> dict[str, int | None]:
            grades_payload = payload.get("grades", payload)
            if not isinstance(grades_payload, dict):
                raise LLMParseError("verifier payload must include an object under 'grades'")
            out: dict[str, int | None] = {}
            for criterion in rubric.criteria:
                raw_value = grades_payload.get(criterion.criterion_id)
                out[criterion.criterion_id] = normalize_grade(raw_value)
            return out

        return self._request_validated(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parser=parse,
        )


class LLMPreferenceJudge(LLMComponentBase):
    """Compares two responses using LLM."""

    def __init__(
        self,
        requester: JsonRequester,
        *,
        model: str,
        max_retries: int,
        prompt_repository: PromptRepository | None = None,
    ) -> None:
        super().__init__(
            requester,
            model=model,
            max_retries=max_retries,
            prompt_repository=prompt_repository,
        )

    def compare(
        self,
        prompt: str,
        left: ResponseCandidate,
        right: ResponseCandidate,
    ) -> int:
        context = {
            "prompt_json": json.dumps(prompt, ensure_ascii=False),
            "left_json": json.dumps(candidate_snapshot(left), ensure_ascii=False),
            "right_json": json.dumps(candidate_snapshot(right), ensure_ascii=False),
        }

        system_prompt, user_prompt = self._render_with_fallback(
            template_id="judge",
            context=context,
            fallback_system=prompt_constants.JUDGE_SYSTEM,
            fallback_template=prompt_constants.JUDGE_USER_TEMPLATE,
        )

        def parse(payload: dict[str, Any]) -> int:
            if "preference" not in payload:
                raise LLMParseError("judge payload must include 'preference'")
            return normalize_preference(payload["preference"])

        return self._request_validated(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parser=parse,
        )
