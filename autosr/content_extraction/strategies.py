from __future__ import annotations

import re
from typing import Protocol

from .parsers import extract_regex_segments, extract_tagged_segments


class ContentExtractor(Protocol):
    def extract(self, text: str) -> str:
        """Extract content from text."""


class TagExtractor:
    def __init__(
        self,
        tag_name: str,
        *,
        join_multiple: str = "\n\n",
        case_sensitive: bool = False,
    ) -> None:
        self._tag_name = tag_name
        self._join_multiple = join_multiple
        self._case_sensitive = case_sensitive

    def extract(self, text: str) -> str:
        cleaned_matches = extract_tagged_segments(
            text,
            tag_name=self._tag_name,
            case_sensitive=self._case_sensitive,
        )
        if not cleaned_matches:
            return text
        return self._join_multiple.join(cleaned_matches)

    @property
    def tag_name(self) -> str:
        return self._tag_name


class RegexExtractor:
    def __init__(
        self,
        pattern: str,
        *,
        group: int = 1,
        join_multiple: str = "\n\n",
        flags: int = re.DOTALL,
    ) -> None:
        self._pattern = re.compile(pattern, flags)
        self._group = group
        self._join_multiple = join_multiple

    def extract(self, text: str) -> str:
        cleaned = extract_regex_segments(
            text,
            compiled_pattern=self._pattern,
            group=self._group,
        )
        if not cleaned:
            return text
        return self._join_multiple.join(cleaned)


class IdentityExtractor:
    def extract(self, text: str) -> str:
        return text


class AnswerExtractor:
    """Extract answer-visible content from model outputs.

    Strategy:
    1) Prefer explicit <answer>...</answer> blocks when present.
    2) Otherwise strip <think>...</think> blocks and keep the remainder.
    3) If stripping removes everything, return the original text as fallback.
    """

    def __init__(self, *, join_multiple: str = "\n\n") -> None:
        self._join_multiple = join_multiple
        self._think_pattern = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

    def extract(self, text: str) -> str:
        answer_segments = extract_tagged_segments(
            text,
            tag_name="answer",
            case_sensitive=False,
        )
        if answer_segments:
            return self._join_multiple.join(answer_segments)

        stripped = self._think_pattern.sub("", text).strip()
        if stripped:
            return stripped
        return text


identity_extractor = IdentityExtractor()
