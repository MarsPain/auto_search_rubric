from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Mapping


def _clamp_binary(value: int | float | None) -> int | None:
    if value is None:
        return None
    return 1 if float(value) >= 0.5 else 0


@dataclass(slots=True)
class Criterion:
    criterion_id: str
    text: str
    weight: float = 1.0
    criterion_type: str | None = None
    check_points: list[str] = field(default_factory=list)
    positive_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.criterion_id.strip():
            raise ValueError("criterion_id must not be empty")
        if not self.text.strip():
            raise ValueError("criterion text must not be empty")
        if self.weight < 0:
            raise ValueError("criterion weight must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "text": self.text,
            "weight": self.weight,
            "criterion_type": self.criterion_type,
            "check_points": self.check_points,
            "positive_examples": self.positive_examples,
            "negative_examples": self.negative_examples,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Criterion":
        # Check required fields and provide meaningful error messages
        if "criterion_id" not in raw:
            raise ValueError("criterion is missing required field 'criterion_id'")
        if "text" not in raw:
            raise ValueError(f"criterion '{raw.get('criterion_id', '<unknown>')}' is missing required field 'text'")
        
        return cls(
            criterion_id=str(raw["criterion_id"]),
            text=str(raw["text"]),
            weight=float(raw.get("weight", 1.0)),
            criterion_type=raw.get("criterion_type"),
            check_points=list(raw.get("check_points", [])),
            positive_examples=list(raw.get("positive_examples", [])),
            negative_examples=list(raw.get("negative_examples", [])),
        )


@dataclass(slots=True)
class GradingProtocol:
    output_format: str = "json"
    allow_na: bool = True
    num_votes: int = 3
    vote_method: str = "majority"
    strict_json: bool = True

    def __post_init__(self) -> None:
        if self.num_votes < 1:
            raise ValueError("num_votes must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_format": self.output_format,
            "allow_na": self.allow_na,
            "num_votes": self.num_votes,
            "vote_method": self.vote_method,
            "strict_json": self.strict_json,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GradingProtocol":
        return cls(
            output_format=str(raw.get("output_format", "json")),
            allow_na=bool(raw.get("allow_na", True)),
            num_votes=int(raw.get("num_votes", 3)),
            vote_method=str(raw.get("vote_method", "majority")),
            strict_json=bool(raw.get("strict_json", True)),
        )


@dataclass(slots=True)
class Rubric:
    rubric_id: str
    criteria: list[Criterion]
    grading_protocol: GradingProtocol = field(default_factory=GradingProtocol)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rubric_id.strip():
            raise ValueError("rubric_id must not be empty")
        if not self.criteria:
            raise ValueError("rubric must include at least one criterion")
        # Validate uniqueness of criterion_id
        seen_ids = set()
        for criterion in self.criteria:
            if criterion.criterion_id in seen_ids:
                raise ValueError(f"duplicate criterion_id: {criterion.criterion_id}")
            seen_ids.add(criterion.criterion_id)

    def normalized_weights(self) -> dict[str, float]:
        total = sum(c.weight for c in self.criteria)
        if total <= 0:
            uniform = 1.0 / len(self.criteria)
            return {c.criterion_id: uniform for c in self.criteria}
        return {c.criterion_id: c.weight / total for c in self.criteria}

    def score_from_grades(self, grades: Mapping[str, int | float | None]) -> float:
        weights = self.normalized_weights()
        numerator = 0.0
        denominator = 0.0
        for criterion in self.criteria:
            grade = _clamp_binary(grades.get(criterion.criterion_id))
            if grade is None:
                if self.grading_protocol.allow_na:
                    continue
                grade = 0
            weight = weights[criterion.criterion_id]
            numerator += weight * float(grade)
            denominator += weight

        if denominator <= 0:
            return 0.0
        return numerator / denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "rubric_id": self.rubric_id,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "grading_protocol": self.grading_protocol.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Rubric":
        criteria = [Criterion.from_dict(item) for item in raw["criteria"]]
        protocol = GradingProtocol.from_dict(raw.get("grading_protocol", {}))
        return cls(
            rubric_id=str(raw["rubric_id"]),
            criteria=criteria,
            grading_protocol=protocol,
            metadata=dict(raw.get("metadata", {})),
        )

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ResponseCandidate:
    candidate_id: str
    text: str
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResponseCandidate":
        return cls(
            candidate_id=str(raw["candidate_id"]),
            text=str(raw["text"]),
            source=str(raw.get("source", "unknown")),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(slots=True)
class PromptExample:
    prompt_id: str
    prompt: str
    candidates: list[ResponseCandidate]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt_id.strip():
            raise ValueError("prompt_id must not be empty")
        if not self.prompt.strip():
            raise ValueError("prompt text must not be empty")
        if len(self.candidates) < 2:
            raise ValueError("each prompt needs at least 2 candidate responses")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PromptExample":
        return cls(
            prompt_id=str(raw["prompt_id"]),
            prompt=str(raw["prompt"]),
            candidates=[ResponseCandidate.from_dict(x) for x in raw["candidates"]],
            metadata=dict(raw.get("metadata", {})),
        )

