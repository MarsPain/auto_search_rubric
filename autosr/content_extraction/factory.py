from __future__ import annotations

import re

from ..interfaces import Verifier
from .strategies import (
    AnswerExtractor,
    ContentExtractor,
    RegexExtractor,
    TagExtractor,
    identity_extractor,
)
from .use_cases import ContentExtractingVerifier


def create_content_extractor(
    strategy: str | None,
    **kwargs,
) -> ContentExtractor:
    if strategy is None or strategy == "identity":
        return identity_extractor

    strategy = strategy.lower()

    if strategy in ("tag", "xml"):
        tag_name = kwargs.get("tag_name")
        if not tag_name:
            raise ValueError("TagExtractor requires 'tag_name' argument")
        return TagExtractor(
            tag_name=tag_name,
            join_multiple=kwargs.get("join_multiple", "\n\n"),
            case_sensitive=kwargs.get("case_sensitive", False),
        )

    if strategy == "regex":
        pattern = kwargs.get("pattern")
        if not pattern:
            raise ValueError("RegexExtractor requires 'pattern' argument")
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
    *,
    candidate_strategy: str | None = None,
    candidate_join_multiple: str = "\n\n",
    **kwargs,
) -> Verifier:
    prompt_extractor = create_content_extractor(strategy, **kwargs)
    candidate_extractor = create_candidate_text_extractor(
        candidate_strategy,
        join_multiple=candidate_join_multiple,
    )

    if prompt_extractor is identity_extractor and candidate_extractor is identity_extractor:
        return verifier
    return ContentExtractingVerifier(
        verifier,
        prompt_extractor,
        candidate_extractor=candidate_extractor,
    )


def create_candidate_text_extractor(
    strategy: str | None,
    **kwargs,
) -> ContentExtractor:
    if strategy is None or strategy == "identity":
        return identity_extractor

    strategy = strategy.lower()
    if strategy == "answer":
        return AnswerExtractor(join_multiple=kwargs.get("join_multiple", "\n\n"))

    raise ValueError(
        f"Unknown candidate extraction strategy: {strategy}. "
        f"Supported: 'answer', 'identity'"
    )
