from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Callable, Mapping, Protocol

from .llm_client import LLMParseError
from .models import GradingProtocol, PromptExample, ResponseCandidate, Rubric

# Import prompt management
from .prompts import constants as prompt_constants
from .prompts.loader import (
    ConstantPromptRepository,
    PromptRepository,
    create_repository,
)

logger = logging.getLogger(__name__)

_REQUIRED_CRITERION_FIELDS = ["criterion_id", "text", "weight"]


def _get_required_fields_json() -> str:
    """Return JSON-encoded required criterion fields."""
    return json.dumps(_REQUIRED_CRITERION_FIELDS)


class JsonRequester(Protocol):
    def request_json(self, *, model: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Call an LLM and return a parsed JSON object."""


class _LLMComponentBase:
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
        # Use provided repository or fall back to constants
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
        """Get prompt configuration from repository."""
        return self._repository.get(template_id, version)


class LLMRubricInitializer(_LLMComponentBase):
    """Initializes rubrics using LLM.
    
    Supports external prompt configuration via prompt_repository parameter.
    """

    def initialize(self, item: PromptExample, *, rng: random.Random) -> Rubric:
        # Build context for template rendering
        candidate_samples = [_candidate_snapshot(c) for c in item.candidates[:4]]
        
        try:
            # Try to use configured template
            config = self._get_prompt_config("rubric_initializer")
            system_prompt, user_prompt = config.render(
                prompt_json=json.dumps(item.prompt, ensure_ascii=False),
                candidate_samples_json=json.dumps(candidate_samples, ensure_ascii=False),
                required_fields_json=_get_required_fields_json(),
            )
        except (KeyError, FileNotFoundError, ValueError) as exc:
            # Fallback to constants (backward compatible)
            logger.debug("Using constant fallback for rubric_initializer: %s", exc)
            system_prompt = prompt_constants.RUBRIC_INITIALIZER_SYSTEM
            user_prompt = prompt_constants.RUBRIC_INITIALIZER_USER_TEMPLATE.format(
                prompt_json=json.dumps(item.prompt, ensure_ascii=False),
                candidate_samples_json=json.dumps(candidate_samples, ensure_ascii=False),
                required_fields_json=_get_required_fields_json(),
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
    """Proposes rubric mutations using LLM.
    
    Supports external prompt configuration via prompt_repository parameter.
    """

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
        # Build context for template rendering
        context = {
            "mode": mode,
            "prompt_json": json.dumps(prompt, ensure_ascii=False),
            "current_rubric_json": json.dumps(rubric.to_dict(), ensure_ascii=False),
            "top_candidate_json": json.dumps(_candidate_snapshot(left), ensure_ascii=False),
            "runner_up_candidate_json": json.dumps(_candidate_snapshot(right), ensure_ascii=False),
            "required_fields_json": _get_required_fields_json(),
        }
        
        try:
            # Try to use configured template
            config = self._get_prompt_config("rubric_proposer")
            system_prompt, user_prompt = config.render(**context)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            # Fallback to constants (backward compatible)
            logger.debug("Using constant fallback for rubric_proposer: %s", exc)
            system_prompt = prompt_constants.RUBRIC_PROPOSER_SYSTEM
            user_prompt = prompt_constants.RUBRIC_PROPOSER_USER_TEMPLATE.format(**context)
        
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
    """Verifies candidate responses using LLM.
    
    Supports external prompt configuration via prompt_repository parameter.
    """

    def grade(
        self,
        prompt: str,
        candidate: ResponseCandidate,
        rubric: Rubric,
        *,
        seed: int,
    ) -> dict[str, int | None]:
        # Build context for template rendering
        criteria_data = [criterion.to_dict() for criterion in rubric.criteria]
        allow_na = "true" if rubric.grading_protocol.allow_na else "false"
        
        context = {
            "seed": seed,
            "prompt_json": json.dumps(prompt, ensure_ascii=False),
            "candidate_json": json.dumps(_candidate_snapshot(candidate), ensure_ascii=False),
            "criteria_json": json.dumps(criteria_data, ensure_ascii=False),
            "allow_na": allow_na,
        }
        
        try:
            # Try to use configured template
            config = self._get_prompt_config("verifier")
            system_prompt, user_prompt = config.render(**context)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            # Fallback to constants (backward compatible)
            logger.debug("Using constant fallback for verifier: %s", exc)
            system_prompt = prompt_constants.VERIFIER_SYSTEM
            user_prompt = prompt_constants.VERIFIER_USER_TEMPLATE.format(**context)

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
    """Compares two responses using LLM.
    
    Supports external prompt configuration via prompt_repository parameter.
    """

    def compare(
        self,
        prompt: str,
        left: ResponseCandidate,
        right: ResponseCandidate,
    ) -> int:
        # Build context for template rendering
        context = {
            "prompt_json": json.dumps(prompt, ensure_ascii=False),
            "left_json": json.dumps(_candidate_snapshot(left), ensure_ascii=False),
            "right_json": json.dumps(_candidate_snapshot(right), ensure_ascii=False),
        }
        
        try:
            # Try to use configured template
            config = self._get_prompt_config("judge")
            system_prompt, user_prompt = config.render(**context)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            # Fallback to constants (backward compatible)
            logger.debug("Using constant fallback for judge: %s", exc)
            system_prompt = prompt_constants.JUDGE_SYSTEM
            user_prompt = prompt_constants.JUDGE_USER_TEMPLATE.format(**context)

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


# Export factory function for convenience
def create_llm_components(
    requester: JsonRequester,
    *,
    model: str,
    max_retries: int = 3,
    prompt_config_path: str | None = None,
):
    """Factory to create all LLM components with optional external configuration.
    
    Args:
        requester: LLM client implementing JsonRequester protocol
        model: Model name to use
        max_retries: Number of retry attempts for failed requests
        prompt_config_path: Optional path to YAML/JSON prompt configs.
                           If None, uses built-in constants.
    
    Returns:
        Tuple of (initializer, proposer, verifier, judge)
    """
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
