"""Reward Harness — automated rubric search and reward model engineering.

This package is the recommended entry point for new code. It re-exports the
public surface of ``autosr`` so that ``reward_harness.*`` and ``autosr.*``
refer to the same objects.

Legacy imports from ``autosr`` remain supported during the migration period.
"""

from __future__ import annotations

# Re-export the entire public surface from autosr so that identity tests
# like ``reward_harness.data_models.Rubric is autosr.data_models.Rubric``
# pass without duplication.
from autosr import (  # noqa: F401
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
