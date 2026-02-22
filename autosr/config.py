"""Configuration classes for organizing CLI parameters by functionality.

This module provides structured configuration objects that group related
parameters together, reducing the parameter explosion in CLI and factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import BackendType, ExtractionStrategy, InitializerStrategy, LLMRole, SearchMode


@dataclass(frozen=True, slots=True)
class LLMBackendConfig:
    """Configuration for LLM backend connection and models.
    
    Groups all parameters related to:
    - API connection (base_url, api_key, timeout)
    - Model selection (default + per-role overrides)
    - Retry behavior
    """
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str | None = None
    timeout: float = 30.0
    max_retries: int = 2
    temperature: float = 0.0
    
    # Model configuration
    default_model: str = "stepfun/step-3.5-flash:free"
    initializer_model: str | None = None
    proposer_model: str | None = None
    verifier_model: str | None = None
    judge_model: str | None = None
    
    def get_model_for_role(self, role: str | LLMRole) -> str:
        """Get the model for a specific role, falling back to default.
        
        Args:
            role: The LLM role (can be string or LLMRole enum)
            
        Returns:
            The model identifier for the given role
        """
        role_key = str(role) if isinstance(role, LLMRole) else role
        role_models = {
            "initializer": self.initializer_model,
            "proposer": self.proposer_model,
            "verifier": self.verifier_model,
            "judge": self.judge_model,
        }
        return role_models.get(role_key, self.default_model) or self.default_model


@dataclass(frozen=True, slots=True)
class SearchAlgorithmConfig:
    """Configuration for search algorithm parameters.
    
    Supports both iterative and evolutionary modes with shared validation.
    Includes advanced selection and mutation strategies for evolutionary mode.
    """
    mode: SearchMode = field(default_factory=lambda: SearchMode.EVOLUTIONARY)
    seed: int = 7
    
    # Iterative mode parameters
    iterations: int = 6
    
    # Evolutionary mode parameters - Basic
    generations: int = 12
    population_size: int = 8
    mutations_per_round: int = 6
    batch_size: int = 3
    survival_fraction: float = 0.2
    elitism_count: int = 2
    stagnation_generations: int = 6
    
    # === Selection Strategy Parameters ===
    # "rank" - Original rank-based selection
    # "tournament" - Tournament selection with configurable size
    # "top_k" - Top-K with diversity protection
    selection_strategy: str = "rank"
    tournament_size: int = 3  # For tournament selection
    tournament_p: float = 0.8  # Probability of selecting best from tournament
    top_k_ratio: float = 0.3  # Ratio of population as elite pool
    diversity_weight: float = 0.3  # Weight for diversity in selection (0-1)
    
    # === Adaptive Mutation Parameters ===
    # "fixed" - Original fixed cycle through modes
    # "success_feedback" - Adapt based on mutation success rate
    # "exploration_decay" - High exploration early, exploitation later
    # "diversity_driven" - Increase weight for modes that improve diversity
    adaptive_mutation: str = "fixed"
    mutation_window_size: int = 10  # History window for tracking
    min_mutation_weight: float = 0.1  # Minimum weight for any mode
    exploration_phase_ratio: float = 0.3  # Ratio of generations for exploration
    diversity_threshold: float = 0.05  # Threshold for diversity-boosting
    
    def __post_init__(self) -> None:
        # Convert string to enum if needed (for backward compatibility)
        if isinstance(self.mode, str):
            object.__setattr__(self, 'mode', SearchMode.from_string(self.mode))
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if self.generations < 1:
            raise ValueError("generations must be >= 1")
        if self.mutations_per_round < 1:
            raise ValueError("mutations_per_round must be >= 1")
        
        # Validate new selection strategy parameters
        if self.selection_strategy not in ("rank", "tournament", "top_k"):
            raise ValueError("selection_strategy must be 'rank', 'tournament', or 'top_k'")
        if self.tournament_size < 2:
            raise ValueError("tournament_size must be >= 2")
        if not 0 < self.tournament_p <= 1:
            raise ValueError("tournament_p must be in (0, 1]")
        if not 0 < self.top_k_ratio <= 1:
            raise ValueError("top_k_ratio must be in (0, 1]")
        if not 0 <= self.diversity_weight <= 1:
            raise ValueError("diversity_weight must be in [0, 1]")
        
        # Validate new adaptive mutation parameters
        if self.adaptive_mutation not in ("fixed", "success_feedback", "exploration_decay", "diversity_driven"):
            raise ValueError("adaptive_mutation must be 'fixed', 'success_feedback', 'exploration_decay', or 'diversity_driven'")
        if self.mutation_window_size < 1:
            raise ValueError("mutation_window_size must be >= 1")
        if not 0 < self.min_mutation_weight <= 1:
            raise ValueError("min_mutation_weight must be in (0, 1]")
        if not 0 < self.exploration_phase_ratio <= 1:
            raise ValueError("exploration_phase_ratio must be in (0, 1]")
        if not 0 <= self.diversity_threshold <= 1:
            raise ValueError("diversity_threshold must be in [0, 1]")

    def to_iterative_kwargs(self) -> dict[str, Any]:
        """Return kwargs for IterativeConfig."""
        return {"iterations": self.iterations, "seed": self.seed}
    
    def to_evolutionary_kwargs(self) -> dict[str, Any]:
        """Return kwargs for EvolutionaryConfig."""
        return {
            "generations": self.generations,
            "population_size": self.population_size,
            "mutations_per_round": self.mutations_per_round,
            "batch_size": self.batch_size,
            "survival_fraction": self.survival_fraction,
            "elitism_count": self.elitism_count,
            "stagnation_generations": self.stagnation_generations,
            "seed": self.seed,
            "selection_strategy": self.selection_strategy,
            "tournament_size": self.tournament_size,
            "tournament_p": self.tournament_p,
            "top_k_ratio": self.top_k_ratio,
            "diversity_weight": self.diversity_weight,
            "adaptive_mutation": self.adaptive_mutation,
            "mutation_window_size": self.mutation_window_size,
            "min_mutation_weight": self.min_mutation_weight,
            "exploration_phase_ratio": self.exploration_phase_ratio,
            "diversity_threshold": self.diversity_threshold,
        }
    
    def is_iterative(self) -> bool:
        """Check if the search mode is iterative."""
        return self.mode is SearchMode.ITERATIVE
    
    def is_evolutionary(self) -> bool:
        """Check if the search mode is evolutionary."""
        return self.mode is SearchMode.EVOLUTIONARY


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    """Configuration for objective function parameters."""
    tail_fraction: float = 0.25
    lambda_var: float = 0.2  # Penalty coefficient for tail variance
    mu_diverse: float = 0.25  # Bonus for cross-source tail alignment
    
    # Pair budget configuration for successive halving
    pair_budget_small: int = 8
    pair_budget_medium: int = 24
    pair_budget_full: int = 0  # 0 means unlimited
    
    tie_tolerance: float = 1e-8
    
    def __post_init__(self) -> None:
        if not 0 < self.tail_fraction <= 1:
            raise ValueError("tail_fraction must be in (0, 1]")
        if self.pair_budget_small < 0 or self.pair_budget_medium < 0 or self.pair_budget_full < 0:
            raise ValueError("pair budgets must be >= 0")


# Backward compatibility alias. New code should use ObjectiveConfig.
ObjectiveFunctionConfig = ObjectiveConfig


@dataclass(frozen=True, slots=True)
class InitializerStrategyConfig:
    """Configuration for rubric initialization strategy."""
    strategy: InitializerStrategy = field(default_factory=lambda: InitializerStrategy.BACKEND)
    preset_rubrics_path: str | None = None
    strict: bool = False  # Require every prompt_id to have a preset rubric
    
    def __post_init__(self) -> None:
        # Convert string to enum if needed (for backward compatibility)
        if isinstance(self.strategy, str):
            object.__setattr__(self, 'strategy', InitializerStrategy.from_string(self.strategy))
        if self.strategy is InitializerStrategy.PRESET and not self.preset_rubrics_path:
            raise ValueError("preset strategy requires preset_rubrics_path")


@dataclass(frozen=True, slots=True)
class ContentExtractionConfig:
    """Configuration for content extraction from verifier input."""
    strategy: ExtractionStrategy = field(default_factory=lambda: ExtractionStrategy.IDENTITY)
    tag_name: str = "content"
    pattern: str | None = None
    join_separator: str = "\n\n"
    
    def __post_init__(self) -> None:
        # Convert string to enum if needed (for backward compatibility)
        if isinstance(self.strategy, str):
            object.__setattr__(self, 'strategy', ExtractionStrategy.from_string(self.strategy))
        if self.strategy is ExtractionStrategy.REGEX and not self.pattern:
            raise ValueError("regex strategy requires pattern")


@dataclass(frozen=True, slots=True)
class VerifierConfig:
    """Configuration for verifier behavior."""
    noise: float = 0.08  # Noise level for heuristic verifier


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Top-level configuration aggregating all sub-configurations.
    
    This is the single object passed to ComponentFactory.
    """
    backend: BackendType = field(default_factory=lambda: BackendType.AUTO)
    llm: LLMBackendConfig = field(default_factory=LLMBackendConfig)
    search: SearchAlgorithmConfig = field(default_factory=SearchAlgorithmConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    initializer: InitializerStrategyConfig = field(default_factory=InitializerStrategyConfig)
    extraction: ContentExtractionConfig = field(default_factory=ContentExtractionConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    
    def __post_init__(self) -> None:
        # Convert string to enum if needed (for backward compatibility)
        if isinstance(self.backend, str):
            object.__setattr__(self, 'backend', BackendType.from_string(self.backend))
    
    def resolve_backend(self) -> BackendType:
        """Resolve auto backend to concrete implementation.
        
        Returns:
            The resolved backend type (MOCK or LLM)
        """
        if self.backend is BackendType.AUTO:
            return BackendType.LLM if self.llm.api_key else BackendType.MOCK
        if self.backend is BackendType.LLM and not self.llm.api_key:
            raise ValueError(
                "LLM backend requires an API key. "
                "Set LLM_API_KEY or use --api-key-env to choose another variable."
            )
        return self.backend
