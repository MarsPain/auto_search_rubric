from __future__ import annotations

from ..interfaces import Verifier
from ..data_models import ResponseCandidate, Rubric
from .strategies import ContentExtractor, identity_extractor


class ContentExtractingVerifier:
    def __init__(
        self,
        inner_verifier: Verifier,
        extractor: ContentExtractor,
        *,
        candidate_extractor: ContentExtractor | None = None,
    ) -> None:
        self._inner = inner_verifier
        self._extractor = extractor
        self._candidate_extractor = candidate_extractor or identity_extractor

    def grade(
        self,
        prompt: str,
        candidate: ResponseCandidate,
        rubric: Rubric,
        *,
        seed: int,
    ) -> dict[str, int | None]:
        extracted = self._extractor.extract(prompt)
        extracted_candidate = self._extract_candidate(candidate)
        return self._inner.grade(extracted, extracted_candidate, rubric, seed=seed)

    def _extract_candidate(self, candidate: ResponseCandidate) -> ResponseCandidate:
        extracted_text = self._candidate_extractor.extract(candidate.text)
        if extracted_text == candidate.text:
            return candidate
        return ResponseCandidate(
            candidate_id=candidate.candidate_id,
            text=extracted_text,
            source=candidate.source,
            metadata=dict(candidate.metadata),
        )

    @property
    def inner_verifier(self) -> Verifier:
        return self._inner

    @property
    def extractor(self) -> ContentExtractor:
        return self._extractor

    @property
    def candidate_extractor(self) -> ContentExtractor:
        return self._candidate_extractor
