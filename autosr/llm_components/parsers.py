from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping

from ..exceptions import LLMParseError
from ..data_models import GradingProtocol, Rubric

logger = logging.getLogger("autosr.llm_components")


def payload_to_rubric(
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
    normalized["criteria"] = normalize_criteria(criteria)
    if not normalized["criteria"]:
        logger.error("All criteria were invalid after normalization: %s", json.dumps(payload, ensure_ascii=False, indent=2))
        raise LLMParseError("rubric payload criteria are invalid")
    normalized.setdefault("rubric_id", fallback_rubric_id)
    normalized.setdefault("grading_protocol", fallback_protocol.to_dict())
    try:
        return Rubric.from_dict(normalized)
    except Exception as exc:  # noqa: BLE001 - schema failures are converted to parse errors.
        raise LLMParseError(f"invalid rubric schema: {exc}") from exc


def normalize_criteria(raw_criteria: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw_criteria, start=1):
        criterion = normalize_one_criterion(entry, idx)
        if criterion is None:
            logger.warning("Dropping invalid criterion entry at position=%d", idx)
            continue
        normalized.append(criterion)
    return normalized


def normalize_one_criterion(entry: Any, index: int) -> dict[str, Any] | None:
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
        text = derive_criterion_text(criterion, criterion_id)
        logger.warning("Criterion %s missing text; synthesized text=%r", criterion_id, text)
    criterion["text"] = text
    return criterion


def derive_criterion_text(criterion: dict[str, Any], criterion_id: str) -> str:
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


def candidate_snapshot(candidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "text": candidate.text,
        "metadata": candidate.metadata,
    }


def normalize_grade(value: Any) -> int | None:
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


def normalize_preference(value: Any) -> int:
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
