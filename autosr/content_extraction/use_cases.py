from __future__ import annotations

from ..interfaces import Verifier
from ..models import ResponseCandidate, Rubric
from .strategies import ContentExtractor


class ContentExtractingVerifier:
    def __init__(
        self,
        inner_verifier: Verifier,
        extractor: ContentExtractor,
    ) -> None:
        self._inner = inner_verifier
        self._extractor = extractor

    def grade(
        self,
        prompt: str,
        candidate: ResponseCandidate,
        rubric: Rubric,
        *,
        seed: int,
    ) -> dict[str, int | None]:
        extracted = self._extractor.extract(prompt)
        return self._inner.grade(extracted, candidate, rubric, seed=seed)

    @property
    def inner_verifier(self) -> Verifier:
        return self._inner

    @property
    def extractor(self) -> ContentExtractor:
        return self._extractor
