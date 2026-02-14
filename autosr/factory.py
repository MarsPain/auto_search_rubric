"""Component factory for creating search components based on configuration.

This module centralizes all component instantiation logic, allowing CLI
and other entry points to remain agnostic of implementation details.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .content_extraction import create_verifier_with_extraction
from .interfaces import PreferenceJudge, RubricInitializer, RubricProposer, Verifier
from .llm_client import LLMClient
from .llm_components import (
    LLMPreferenceJudge,
    LLMRubricInitializer,
    LLMRubricProposer,
    LLMVerifier,
)
from .llm_config import LLMConfig, RoleModelConfig
from .mock_components import (
    HeuristicPreferenceJudge,
    HeuristicRubricInitializer,
    HeuristicVerifier,
    PresetRubricInitializer,
    RankPreferenceJudge,
    TemplateProposer,
)
from .models import PromptExample
from .search import EvolutionaryConfig, EvolutionaryRTDSearcher, IterativeConfig, IterativeRTDSearcher
from .types import BackendType, InitializerStrategy, LLMRole, SearchMode

if TYPE_CHECKING:
    from .config import RuntimeConfig

logger = logging.getLogger(__name__)


class ComponentFactory:
    """Factory for creating search components from configuration.
    
    This factory encapsulates all knowledge about:
    - Which components to instantiate based on backend type
    - How to wire dependencies between components
    - How to apply decorators/wrappers (e.g., content extraction)
    
    Usage:
        config = RuntimeConfig(...)
        factory = ComponentFactory(config)
        searcher = factory.create_searcher()
    """
    
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._backend = config.resolve_backend()
        self._llm_client: LLMClient | None = None
    
    def _get_llm_client(self) -> LLMClient:
        """Get or create the shared LLM client."""
        if self._llm_client is None:
            llm_cfg = self.config.llm
            llm_config = LLMConfig(
                base_url=llm_cfg.base_url,
                api_key=llm_cfg.api_key,
                timeout=llm_cfg.timeout,
                max_retries=llm_cfg.max_retries,
                temperature=llm_cfg.temperature,
            )
            self._llm_client = LLMClient(llm_config)
        return self._llm_client
    
    def _has_rank_for_all_candidates(self, prompts: list[PromptExample]) -> bool:
        """Check if all candidates across all prompts have metadata.rank defined."""
        for prompt in prompts:
            for candidate in prompt.candidates:
                if candidate.metadata.get("rank") is None:
                    return False
        return True
    
    def create_proposer(self) -> RubricProposer:
        """Create the rubric proposer component."""
        if self._backend is BackendType.MOCK:
            return TemplateProposer()
        
        llm_cfg = self.config.llm
        return LLMRubricProposer(
            self._get_llm_client(),
            model=llm_cfg.get_model_for_role(LLMRole.PROPOSER),
            max_retries=llm_cfg.max_retries,
        )
    
    def create_verifier(self) -> Verifier:
        """Create the verifier component (without extraction wrapper)."""
        if self._backend is BackendType.MOCK:
            return HeuristicVerifier(noise=self.config.verifier.noise)
        
        llm_cfg = self.config.llm
        return LLMVerifier(
            self._get_llm_client(),
            model=llm_cfg.get_model_for_role(LLMRole.VERIFIER),
            max_retries=llm_cfg.max_retries,
        )
    
    def create_judge(self, prompts: list[PromptExample]) -> PreferenceJudge:
        """Create the preference judge component.
        
        Auto-selects RankPreferenceJudge if all candidates have rank metadata.
        """
        # Auto-select based on dataset characteristics
        if self._has_rank_for_all_candidates(prompts):
            logger.info("Detected metadata.rank in all candidates; using RankPreferenceJudge")
            return RankPreferenceJudge()
        
        if self._backend is BackendType.MOCK:
            return HeuristicPreferenceJudge()
        
        llm_cfg = self.config.llm
        return LLMPreferenceJudge(
            self._get_llm_client(),
            model=llm_cfg.get_model_for_role(LLMRole.JUDGE),
            max_retries=llm_cfg.max_retries,
        )
    
    def create_base_initializer(self) -> RubricInitializer:
        """Create the base initializer (without preset wrapper)."""
        if self._backend is BackendType.MOCK:
            return HeuristicRubricInitializer()
        
        llm_cfg = self.config.llm
        return LLMRubricInitializer(
            self._get_llm_client(),
            model=llm_cfg.get_model_for_role(LLMRole.INITIALIZER),
            max_retries=llm_cfg.max_retries,
        )
    
    def create_initializer(self) -> RubricInitializer:
        """Create the initializer with preset wrapper if configured."""
        base_initializer = self.create_base_initializer()
        init_cfg = self.config.initializer
        
        if init_cfg.strategy is InitializerStrategy.PRESET:
            from .io_utils import load_initial_rubrics
            
            preset_rubrics = load_initial_rubrics(init_cfg.preset_rubrics_path)
            return PresetRubricInitializer(
                preset_rubrics,
                fallback_initializer=base_initializer,
                strict=init_cfg.strict,
            )
        
        return base_initializer
    
    def create_verifier_with_extraction(self, prompts: list[PromptExample]) -> Verifier:
        """Create verifier with content extraction wrapper if configured."""
        from .types import ExtractionStrategy
        
        verifier = self.create_verifier()
        ext_cfg = self.config.extraction
        
        if ext_cfg.strategy is ExtractionStrategy.IDENTITY:
            return verifier
        
        extraction_kwargs = {"join_multiple": ext_cfg.join_separator}
        
        if ext_cfg.strategy is ExtractionStrategy.TAG:
            extraction_kwargs["tag_name"] = ext_cfg.tag_name
        elif ext_cfg.strategy is ExtractionStrategy.REGEX:
            extraction_kwargs["pattern"] = ext_cfg.pattern
        
        return create_verifier_with_extraction(
            verifier,
            ext_cfg.strategy.value,
            **extraction_kwargs,
        )
    
    def create_searcher(
        self, prompts: list[PromptExample]
    ) -> IterativeRTDSearcher | EvolutionaryRTDSearcher:
        """Create the search algorithm component with all dependencies."""
        # Create all runtime components
        proposer = self.create_proposer()
        verifier = self.create_verifier_with_extraction(prompts)
        judge = self.create_judge(prompts)
        initializer = self.create_initializer()
        
        objective = self.config.objective
        search_cfg = self.config.search
        
        if search_cfg.mode is SearchMode.ITERATIVE:
            config = IterativeConfig(
                **search_cfg.to_iterative_kwargs(),
                objective=objective,
            )
            return IterativeRTDSearcher(proposer, verifier, judge, initializer, config=config)
        else:
            config = EvolutionaryConfig(
                **search_cfg.to_evolutionary_kwargs(),
                objective=objective,
            )
            return EvolutionaryRTDSearcher(proposer, verifier, judge, initializer, config=config)


def create_components_for_dataset(
    config: RuntimeConfig,
    prompts: list[PromptExample],
) -> tuple[
    RubricProposer,
    Verifier,
    PreferenceJudge,
    RubricInitializer,
]:
    """Create all runtime components for a dataset.
    
    Convenience function for cases where you need direct access to
    individual components rather than the high-level searcher.
    
    Returns:
        Tuple of (proposer, verifier, judge, initializer)
    """
    factory = ComponentFactory(config)
    return (
        factory.create_proposer(),
        factory.create_verifier_with_extraction(prompts),
        factory.create_judge(prompts),
        factory.create_initializer(),
    )
