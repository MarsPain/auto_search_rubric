"""AutoSR — legacy compatibility shim for Reward Harness.

All implementations have moved to ``reward_harness``. Importing from ``autosr``
still works and returns the same objects as ``reward_harness``.
"""

from __future__ import annotations

from reward_harness import (  # noqa: F401
    AnswerExtractor,
    ContentExtractingVerifier,
    IdentityExtractor,
    RegexExtractor,
    TagExtractor,
    create_candidate_text_extractor,
    create_content_extractor,
    create_verifier_with_extraction,
    Criterion,
    GradingProtocol,
    PromptExample,
    ResponseCandidate,
    Rubric,
    EvolutionaryConfig,
    EvolutionaryRTDSearcher,
    IterativeConfig,
    IterativeRTDSearcher,
    SearchResult,
    AdaptiveMutationSchedule,
    BackendType,
    CandidateExtractionStrategy,
    ExtractionStrategy,
    InitializerStrategy,
    LLMRole,
    MutationMode,
    SearchMode,
    SelectionStrategy,
)

__all__ = [
    "AnswerExtractor",
    "ContentExtractingVerifier",
    "IdentityExtractor",
    "RegexExtractor",
    "TagExtractor",
    "create_candidate_text_extractor",
    "create_content_extractor",
    "create_verifier_with_extraction",
    "Criterion",
    "GradingProtocol",
    "PromptExample",
    "ResponseCandidate",
    "Rubric",
    "EvolutionaryConfig",
    "EvolutionaryRTDSearcher",
    "IterativeConfig",
    "IterativeRTDSearcher",
    "SearchResult",
    "AdaptiveMutationSchedule",
    "BackendType",
    "CandidateExtractionStrategy",
    "ExtractionStrategy",
    "InitializerStrategy",
    "LLMRole",
    "MutationMode",
    "SearchMode",
    "SelectionStrategy",
]
