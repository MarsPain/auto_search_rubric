from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Callable, Mapping, Protocol

from .llm_client import LLMParseError
from .models import GradingProtocol, PromptExample, ResponseCandidate, Rubric

logger = logging.getLogger(__name__)

_REQUIRED_CRITERION_FIELDS = ["criterion_id", "text", "weight"]


class JsonRequester(Protocol):
    def request_json(self, *, model: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Call an LLM and return a parsed JSON object."""


class _LLMComponentBase:
    def __init__(self, requester: JsonRequester, *, model: str, max_retries: int) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.requester = requester
        self.model = model
        self.max_retries = max_retries

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


class LLMRubricInitializer(_LLMComponentBase):
    def initialize(self, item: PromptExample, *, rng: random.Random) -> Rubric:
        system_prompt = (
            "You are a rubric designer. Return ONLY one JSON object. "
            "Required top-level keys: rubric_id, criteria, grading_protocol. "
            "Each criterion object MUST include non-empty criterion_id and text, plus weight. "
            "Do not omit required fields. Do not return markdown, comments, or partial patches."
        )
        user_prompt = json.dumps(
            {
                "task": "Create an evaluation rubric for the prompt.",
                "prompt": item.prompt,
                "candidate_samples": [_candidate_snapshot(c) for c in item.candidates[:4]],
                "constraints": {
                    "criteria_count_range": [3, 6],
                    "weights_sum_hint": 1.0,
                    "criterion_required_fields": _REQUIRED_CRITERION_FIELDS,
                    "return_full_rubric": True,
                    "grading_protocol_default": {
                        "num_votes": 1,
                        "allow_na": True,
                        "vote_method": "majority",
                    },
                },
            },
            ensure_ascii=False,
        )
        fallback_id = f"{item.prompt_id}_llm_init_{rng.randint(0, 9999)}"
        return self._request_validated(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parser=lambda payload: _payload_to_rubric(
                payload,
                fallback_rubric_id=fallback_id,
                fallback_protocol=GradingProtocol(num_votes=1, allow_na=True, vote_method="majority"),
            ),
        )


class LLMRubricProposer(_LLMComponentBase):
    def propose(
        self,
        prompt: str,
        left: ResponseCandidate,
        right: ResponseCandidate,
        rubric: Rubric,
        *,
        mode: str,
        rng: random.Random,
    ) -> Rubric:
        system_prompt = (
            "You mutate rubrics for ranking response quality. Return ONLY one JSON rubric object. "
            "Required top-level keys: rubric_id, criteria; grading_protocol is optional. "
            "Each criterion object MUST include non-empty criterion_id and text, plus weight. "
            "Return a full rubric, not a delta/patch. Keep criteria precise and scoreable."
        )
        user_prompt = json.dumps(
            {
                "task": "Mutate the rubric according to mode while preserving intent.",
                "mode": mode,
                "prompt": prompt,
                "current_rubric": rubric.to_dict(),
                "top_candidate": _candidate_snapshot(left),
                "runner_up_candidate": _candidate_snapshot(right),
                "output_requirements": {
                    "must_return_full_rubric": True,
                    "criterion_required_fields": _REQUIRED_CRITERION_FIELDS,
                    "preserve_criterion_text_even_if_unchanged": True,
                },
            },
            ensure_ascii=False,
        )
        fallback_id = f"{rubric.rubric_id}_{mode}_{rng.randint(0, 9999)}"
        return self._request_validated(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parser=lambda payload: _payload_to_rubric(
                payload,
                fallback_rubric_id=fallback_id,
                fallback_protocol=rubric.grading_protocol,
            ),
        )


class LLMVerifier(_LLMComponentBase):
    def grade(
        self,
        prompt: str,
        candidate: ResponseCandidate,
        rubric: Rubric,
        *,
        seed: int,
    ) -> dict[str, int | None]:
        system_prompt = (
            "You are a strict evaluator. Return ONLY JSON: "
            '{"grades": {"criterion_id": 0|1|null, ...}}.'
        )
        user_prompt = json.dumps(
            {
                "seed": seed,
                "prompt": prompt,
                "candidate": _candidate_snapshot(candidate),
                "criteria": [criterion.to_dict() for criterion in rubric.criteria],
                "allow_na": rubric.grading_protocol.allow_na,
            },
            ensure_ascii=False,
        )

        def parse(payload: dict[str, Any]) -> dict[str, int | None]:
            grades_payload = payload.get("grades", payload)
            if not isinstance(grades_payload, dict):
                raise LLMParseError("verifier payload must include an object under 'grades'")
            out: dict[str, int | None] = {}
            for criterion in rubric.criteria:
                raw_value = grades_payload.get(criterion.criterion_id)
                out[criterion.criterion_id] = _normalize_grade(raw_value)
            return out

        return self._request_validated(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parser=parse,
        )


class LLMPreferenceJudge(_LLMComponentBase):
    def compare(
        self,
        prompt: str,
        left: ResponseCandidate,
        right: ResponseCandidate,
    ) -> int:
        system_prompt = (
            "Compare two responses and return ONLY JSON: "
            '{"preference": 1|-1|0} where 1 means left is preferred.'
        )
        user_prompt = json.dumps(
            {
                "prompt": prompt,
                "left": _candidate_snapshot(left),
                "right": _candidate_snapshot(right),
            },
            ensure_ascii=False,
        )

        def parse(payload: dict[str, Any]) -> int:
            if "preference" not in payload:
                raise LLMParseError("judge payload must include 'preference'")
            return _normalize_preference(payload["preference"])

        return self._request_validated(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parser=parse,
        )


def _payload_to_rubric(
    payload: dict[str, Any],
    *,
    fallback_rubric_id: str,
    fallback_protocol: GradingProtocol,
) -> Rubric:
    rubric_payload = payload.get("rubric", payload)
    if not isinstance(rubric_payload, dict):
        logger.error("Invalid rubric payload (not a dict): %s", json.dumps(payload, ensure_ascii=False, indent=2))
        raise LLMParseError("rubric payload must be an object")

    criteria = rubric_payload.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        logger.error("Missing criteria in rubric payload: %s", json.dumps(payload, ensure_ascii=False, indent=2))
        raise LLMParseError("rubric payload must include a non-empty criteria list")

    normalized = dict(rubric_payload)
    normalized["criteria"] = _normalize_criteria(criteria)
    if not normalized["criteria"]:
        logger.error("All criteria were invalid after normalization: %s", json.dumps(payload, ensure_ascii=False, indent=2))
        raise LLMParseError("rubric payload criteria are invalid")
    normalized.setdefault("rubric_id", fallback_rubric_id)
    normalized.setdefault("grading_protocol", fallback_protocol.to_dict())
    try:
        return Rubric.from_dict(normalized)
    except Exception as exc:
        # logger.error(
        #     "Failed to parse rubric from payload: %s\nFull payload: %s",
        #     exc,
        #     json.dumps(payload, ensure_ascii=False, indent=2)
        # )
        raise LLMParseError(f"invalid rubric schema: {exc}") from exc


def _normalize_criteria(raw_criteria: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw_criteria, start=1):
        criterion = _normalize_one_criterion(entry, idx)
        if criterion is None:
            logger.warning("Dropping invalid criterion entry at position=%d", idx)
            continue
        normalized.append(criterion)
    return normalized


def _normalize_one_criterion(entry: Any, index: int) -> dict[str, Any] | None:
    if isinstance(entry, Mapping):
        criterion = dict(entry)
    elif isinstance(entry, str):
        token = entry.strip()
        if not token:
            return None
        criterion = {"criterion_id": f"criterion_{index}", "text": token}
    else:
        return None

    raw_id = criterion.get("criterion_id")
    criterion_id = str(raw_id).strip() if raw_id is not None else ""
    if not criterion_id:
        criterion_id = f"criterion_{index}"
        logger.warning("Criterion missing criterion_id; using fallback id=%s", criterion_id)
    criterion["criterion_id"] = criterion_id

    raw_text = criterion.get("text")
    text = str(raw_text).strip() if raw_text is not None else ""
    if not text:
        text = _derive_criterion_text(criterion, criterion_id)
        logger.warning("Criterion %s missing text; synthesized text=%r", criterion_id, text)
    criterion["text"] = text
    return criterion


def _derive_criterion_text(criterion: dict[str, Any], criterion_id: str) -> str:
    for key in ("description", "criterion_type", "title", "name"):
        value = criterion.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    check_points = criterion.get("check_points")
    if isinstance(check_points, list):
        for item in check_points:
            if isinstance(item, str) and item.strip():
                return item.strip()

    readable = re.sub(r"[_\-]+", " ", criterion_id).strip()
    return readable or criterion_id


def _candidate_snapshot(candidate: ResponseCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "text": candidate.text,
        "metadata": candidate.metadata,
    }


def _normalize_grade(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"n/a", "na", "none", "null"}:
            return None
        if token in {"1", "true", "yes", "pass"}:
            return 1
        if token in {"0", "false", "no", "fail"}:
            return 0
    raise LLMParseError(f"unsupported grade value: {value!r}")


def _normalize_preference(value: Any) -> int:
    if isinstance(value, bool):
        raise LLMParseError("preference must be one of 1, 0, -1")
    if isinstance(value, (int, float)):
        normalized = int(value)
        if normalized in {-1, 0, 1}:
            return normalized
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "left"}:
            return 1
        if token in {"-1", "right"}:
            return -1
        if token in {"0", "tie", "equal"}:
            return 0
    raise LLMParseError("preference must be one of 1, 0, -1")
