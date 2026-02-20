"""Example: Advanced Evolutionary Search Configuration

This example demonstrates how to use the new selection strategies
and adaptive mutation schedules for improved diversity and convergence.
"""

from autosr.search import EvolutionaryConfig, SelectionStrategy, AdaptiveMutationSchedule

# =============================================================================
# Example 1: Tournament Selection with Success Feedback
# =============================================================================
# Tournament selection provides balanced pressure between quality and diversity.
# Success feedback adapts mutation mode weights based on historical performance.

tournament_success_config = EvolutionaryConfig(
    population_size=12,
    generations=30,
    mutations_per_round=8,
    # Selection strategy
    selection_strategy=SelectionStrategy.TOURNAMENT,
    tournament_size=3,          # Small tournament = more diversity
    tournament_p=0.8,           # 80% chance to pick best from tournament
    # Adaptive mutation
    adaptive_mutation=AdaptiveMutationSchedule.SUCCESS_FEEDBACK,
    mutation_window_size=15,    # Track last 15 mutations per mode
    min_mutation_weight=0.1,    # Ensure no mode is completely ignored
)

print("Configuration 1: Tournament + Success Feedback")
print(f"  Selection: {tournament_success_config.selection_strategy.name}")
print(f"  Tournament size: {tournament_success_config.tournament_size}")
print(f"  Adaptive mutation: {tournament_success_config.adaptive_mutation.name}")
print()

# =============================================================================
# Example 2: Top-K Selection with Diversity Protection
# =============================================================================
# Top-K with diversity protection explicitly considers diversity when selecting
# parents, preventing premature convergence.

topk_diversity_config = EvolutionaryConfig(
    population_size=16,
    generations=25,
    mutations_per_round=10,
    # Selection strategy with diversity protection
    selection_strategy=SelectionStrategy.TOP_K,
    top_k_ratio=0.4,            # Consider top 40% as elite pool
    diversity_weight=0.4,       # 40% weight on diversity, 60% on quality
    # Diversity-driven mutation adapts to population diversity
    adaptive_mutation=AdaptiveMutationSchedule.DIVERSITY_DRIVEN,
    diversity_threshold=0.05,   # Boost diversity when below 5%
)

print("Configuration 2: Top-K + Diversity Driven")
print(f"  Selection: {topk_diversity_config.selection_strategy.name}")
print(f"  Top-K ratio: {topk_diversity_config.top_k_ratio}")
print(f"  Diversity weight: {topk_diversity_config.diversity_weight}")
print(f"  Adaptive mutation: {topk_diversity_config.adaptive_mutation.name}")
print()

# =============================================================================
# Example 3: Exploration Decay Schedule
# =============================================================================
# Exploration decay favors diverse mutations early, then shifts to refinement.
# Good for problems where early exploration and late exploitation are key.

exploration_decay_config = EvolutionaryConfig(
    population_size=10,
    generations=40,
    mutations_per_round=6,
    selection_strategy=SelectionStrategy.TOURNAMENT,
    tournament_size=4,          # Larger tournament = more exploitation
    # Exploration decay schedule
    adaptive_mutation=AdaptiveMutationSchedule.EXPLORATION_DECAY,
    exploration_phase_ratio=0.3,  # First 30% generations favor exploration
)

print("Configuration 3: Tournament + Exploration Decay")
print(f"  Selection: {exploration_decay_config.selection_strategy.name}")
print(f"  Adaptive mutation: {exploration_decay_config.adaptive_mutation.name}")
print(f"  Exploration phase: {exploration_decay_config.exploration_phase_ratio * 100:.0f}%")
print()

# =============================================================================
# Example 4: Original Behavior (for comparison)
# =============================================================================

original_config = EvolutionaryConfig(
    population_size=8,
    generations=20,
    mutations_per_round=6,
    selection_strategy=SelectionStrategy.RANK,  # Original fixed rank selection
    adaptive_mutation=AdaptiveMutationSchedule.FIXED,  # Original fixed cycle
)

print("Configuration 4: Original (Rank + Fixed)")
print(f"  Selection: {original_config.selection_strategy.name}")
print(f"  Adaptive mutation: {original_config.adaptive_mutation.name}")
print()

# =============================================================================
# Usage in code
# =============================================================================

print("Usage Example:")
print("""
from autosr.factory import ComponentFactory
from autosr.io_utils import load_dataset
from autosr.config import RuntimeConfig

# Load dataset
prompts = load_dataset("dataset.json")

# Create runtime config with advanced evolutionary settings
runtime_config = RuntimeConfig(
    search=SearchAlgorithmConfig(
        mode="evolutionary",
        generations=30,
        population_size=12,
        selection_strategy="tournament",  # or "rank", "top_k"
        adaptive_mutation="success_feedback",  # or "fixed", "exploration_decay", "diversity_driven"
    )
)

# Create factory and searcher
factory = ComponentFactory(runtime_config)
searcher = factory.create_searcher(prompts)

# Run search
result = searcher.search(prompts)

# Access diagnostics
print(result.diagnostics["selection_strategy"])
print(result.diagnostics["adaptive_mutation"])
print(result.diagnostics["avg_diversity"])
""")
