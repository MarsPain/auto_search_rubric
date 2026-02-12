"""Content extraction utilities for prompt processing.

This module provides pluggable content extraction strategies that can extract
content from prompts using various methods (XML-style tags, regex, etc.).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Protocol

from .interfaces import Verifier
from .models import ResponseCandidate, Rubric


class ContentExtractor(Protocol):
    """Protocol for content extraction strategies.

    Implementations should extract relevant content from a text prompt
    based on specific rules (tags, regex patterns, etc.).
    """

    def extract(self, text: str) -> str:
        """Extract content from text.

        Args:
            text: The input text potentially containing extractable content.

        Returns:
            Extracted content if found, otherwise the original text.
        """
        ...


class TagExtractor:
    """Extract content from XML-style tags.

    Extracts content from specified XML-style tags (e.g., <content>...</content>).
    Supports multiple matches and custom separators.

    Example:
        >>> extractor = TagExtractor(tag_name="content")
        >>> extractor.extract("Prefix<content>Inner</content>Suffix")
        'Inner'
        >>> extractor.extract("<content>A</content><content>B</content>")
        'A\\n\\nB'
    """

    def __init__(
        self,
        tag_name: str,
        *,
        join_multiple: str = "\n\n",
        case_sensitive: bool = False,
    ) -> None:
        """Initialize the tag extractor.

        Args:
            tag_name: The XML-style tag name to extract content from.
            join_multiple: Separator used when joining multiple tag matches.
            case_sensitive: Whether tag matching is case-sensitive.
        """
        self._tag_name = tag_name
        self._join_multiple = join_multiple
        flags = re.DOTALL
        if not case_sensitive:
            flags |= re.IGNORECASE
        self._pattern = re.compile(
            rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>",
            flags,
        )

    def extract(self, text: str) -> str:
        """Extract content from specified tags in the text.

        Args:
            text: The input text.

        Returns:
            Extracted content if tags found, otherwise original text.
        """
        matches = self._pattern.findall(text)
        if not matches:
            return text

        cleaned_matches = [match.strip() for match in matches if match.strip()]
        if not cleaned_matches:
            return text

        return self._join_multiple.join(cleaned_matches)

    @property
    def tag_name(self) -> str:
        """Get the configured tag name."""
        return self._tag_name


class RegexExtractor:
    """Extract content using regex patterns.

    Uses a regex pattern with capture groups to extract content.
    By default uses the first capture group.

    Example:
        >>> extractor = RegexExtractor(r"Content:\\s*(.+?)(?:\\n|$)")
        >>> extractor.extract("Title\\nContent: Hello World\\nFooter")
        'Hello World'
    """

    def __init__(
        self,
        pattern: str,
        *,
        group: int = 1,
        join_multiple: str = "\n\n",
        flags: int = re.DOTALL,
    ) -> None:
        """Initialize the regex extractor.

        Args:
            pattern: The regex pattern with capture groups.
            group: The capture group index to extract (default: 1).
            join_multiple: Separator for multiple matches.
            flags: Regex flags (default: DOTALL for multiline matching).
        """
        self._pattern = re.compile(pattern, flags)
        self._group = group
        self._join_multiple = join_multiple

    def extract(self, text: str) -> str:
        """Extract content using the regex pattern.

        Args:
            text: The input text.

        Returns:
            Extracted content if matches found, otherwise original text.
        """
        matches = self._pattern.findall(text)
        if not matches:
            return text

        # Handle tuple matches (multiple groups)
        if matches and isinstance(matches[0], tuple):
            cleaned = [m[self._group - 1].strip() for m in matches if len(m) >= self._group]
        else:
            cleaned = [m.strip() for m in matches if m.strip()]

        if not cleaned:
            return text

        return self._join_multiple.join(cleaned)


class IdentityExtractor:
    """Passthrough extractor that returns text unchanged.

    Used as a no-op extractor when no extraction is needed.
    """

    def extract(self, text: str) -> str:
        """Return text unchanged."""
        return text


# Global identity extractor instance
_identity_extractor = IdentityExtractor()


class ContentExtractingVerifier:
    """Wrapper verifier that extracts content before passing to inner verifier.

    This verifier wraps an inner verifier and applies a content extraction
    strategy to the prompt before grading.

    Example:
        >>> inner = LLMVerifier(...)
        >>> extractor = TagExtractor("通话内容")
        >>> verifier = ContentExtractingVerifier(inner, extractor)
        >>> grades = verifier.grade(prompt, candidate, rubric, seed=42)
    """

    def __init__(
        self,
        inner_verifier: Verifier,
        extractor: ContentExtractor,
    ) -> None:
        """Initialize the content extracting verifier.

        Args:
            inner_verifier: The underlying verifier to wrap.
            extractor: The content extraction strategy to apply.
        """
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
        """Grade candidate after extracting content from prompt.

        Args:
            prompt: The full prompt text.
            candidate: The candidate response to evaluate.
            rubric: The rubric to use for grading.
            seed: Random seed for deterministic grading.

        Returns:
            Dict mapping criterion_id to binary grade (0, 1, or None).
        """
        extracted = self._extractor.extract(prompt)
        # print(f"extracted == {extracted}")
        return self._inner.grade(extracted, candidate, rubric, seed=seed)

    @property
    def inner_verifier(self) -> Verifier:
        """Access the wrapped inner verifier."""
        return self._inner

    @property
    def extractor(self) -> ContentExtractor:
        """Get the configured content extractor."""
        return self._extractor


def create_content_extractor(
    strategy: str | None,
    **kwargs,
) -> ContentExtractor:
    """Factory function to create content extractors by strategy name.

    Args:
        strategy: The extraction strategy name:
            - "tag" or "xml": Use TagExtractor (requires 'tag_name')
            - "regex": Use RegexExtractor (requires 'pattern')
            - None or "identity": Use IdentityExtractor (no-op)
        **kwargs: Strategy-specific arguments.

    Returns:
        Configured ContentExtractor instance.

    Raises:
        ValueError: If strategy name is unknown or required args missing.

    Example:
        >>> # Tag extraction
        >>> extractor = create_content_extractor("tag", tag_name="content")
        >>>
        >>> # Regex extraction
        >>> extractor = create_content_extractor("regex", pattern=r"Data:\\s*(.+)")
        >>>
        >>> # No extraction
        >>> extractor = create_content_extractor(None)
    """
    if strategy is None or strategy == "identity":
        return _identity_extractor

    strategy = strategy.lower()

    if strategy in ("tag", "xml"):
        tag_name = kwargs.get("tag_name")
        if not tag_name:
            raise ValueError(f"TagExtractor requires 'tag_name' argument")
        return TagExtractor(
            tag_name=tag_name,
            join_multiple=kwargs.get("join_multiple", "\n\n"),
            case_sensitive=kwargs.get("case_sensitive", False),
        )

    if strategy == "regex":
        pattern = kwargs.get("pattern")
        if not pattern:
            raise ValueError(f"RegexExtractor requires 'pattern' argument")
        return RegexExtractor(
            pattern=pattern,
            group=kwargs.get("group", 1),
            join_multiple=kwargs.get("join_multiple", "\n\n"),
            flags=kwargs.get("flags", re.DOTALL),
        )

    raise ValueError(
        f"Unknown extraction strategy: {strategy}. "
        f"Supported: 'tag', 'xml', 'regex', 'identity'"
    )


def create_verifier_with_extraction(
    verifier: Verifier,
    strategy: str | None,
    **kwargs,
) -> Verifier:
    """Create a verifier optionally wrapped with content extraction.

    Convenience function that combines extractor creation and verifier wrapping.

    Args:
        verifier: The base verifier to wrap.
        strategy: Extraction strategy name (None for no wrapping).
        **kwargs: Strategy-specific arguments.

    Returns:
        ContentExtractingVerifier if strategy provided, else original verifier.
    """
    if strategy is None or strategy == "identity":
        return verifier
    extractor = create_content_extractor(strategy, **kwargs)
    return ContentExtractingVerifier(verifier, extractor)
