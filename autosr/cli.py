"""CLI entry point for tail-focused rubric search.

This module provides a minimal CLI that delegates all component creation
to ComponentFactory and all configuration to RuntimeConfig.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from .config import (
    ContentExtractionConfig,
    InitializerStrategyConfig,
    LLMBackendConfig,
    ObjectiveFunctionConfig,
    RuntimeConfig,
    SearchAlgorithmConfig,
    VerifierConfig,
)
from .types import BackendType, LLMRole
from .evaluator import RubricEvaluator
from .factory import ComponentFactory
from .io_utils import load_dataset, load_initial_rubrics, save_rubrics
from .models import PromptExample, Rubric

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with logically grouped arguments."""
    parser = argparse.ArgumentParser(
        description="Tail-focused rubric search runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # === Core arguments ===
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON")
    parser.add_argument("--output", required=True, help="Where to save best rubric JSON")
    parser.add_argument(
        "--mode",
        choices=["iterative", "evolutionary"],
        default="evolutionary",
        help="Search algorithm mode",
    )
    
    # === Backend selection ===
    backend_group = parser.add_argument_group("Backend Options")
    backend_group.add_argument(
        "--backend",
        choices=["auto", "mock", "llm"],
        default="auto",
        help="Backend selection (auto uses llm when API key exists, otherwise mock)",
    )
    backend_group.add_argument(
        "--api-key-env",
        default="LLM_API_KEY",
        help="Environment variable containing LLM API key",
    )
    
    # === LLM configuration ===
    llm_group = parser.add_argument_group("LLM Options")
    llm_group.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LLM API base URL")
    llm_group.add_argument("--model-default", default=DEFAULT_MODEL, help="Default LLM model")
    llm_group.add_argument("--model-initializer", default=None, help="Model for initializer")
    llm_group.add_argument("--model-proposer", default=None, help="Model for proposer")
    llm_group.add_argument("--model-verifier", default=None, help="Model for verifier")
    llm_group.add_argument("--model-judge", default=None, help="Model for judge")
    llm_group.add_argument("--llm-timeout", type=float, default=30.0, help="Request timeout (s)")
    llm_group.add_argument("--llm-max-retries", type=int, default=2, help="Retry count")
    
    # === Search algorithm parameters ===
    search_group = parser.add_argument_group("Search Algorithm Options")
    search_group.add_argument("--seed", type=int, default=7, help="Random seed")
    search_group.add_argument("--iterations", type=int, default=6, help="[Iterative] steps")
    search_group.add_argument("--generations", type=int, default=12, help="[Evolutionary] generations")
    search_group.add_argument("--population-size", type=int, default=8, help="[Evolutionary] population")
    search_group.add_argument("--mutations-per-round", type=int, default=6, help="Mutations per round")
    search_group.add_argument("--batch-size", type=int, default=3, help="Prompts per generation")
    
    # === Objective function parameters ===
    objective_group = parser.add_argument_group("Objective Function Options")
    objective_group.add_argument("--tail-fraction", type=float, default=0.25, help="Top fraction for tail")
    objective_group.add_argument("--lambda-var", type=float, default=0.2, help="Variance penalty coefficient")
    objective_group.add_argument("--mu-diverse", type=float, default=0.25, help="Diversity bonus coefficient")
    
    # === Verifier options ===
    verifier_group = parser.add_argument_group("Verifier Options")
    verifier_group.add_argument("--noise", type=float, default=0.08, help="Verifier noise level (mock)")
    
    # === Initializer strategy ===
    init_group = parser.add_argument_group("Initializer Strategy Options")
    init_group.add_argument(
        "--initializer-strategy",
        choices=["backend", "preset"],
        default="backend",
        help="Strategy for initial rubrics",
    )
    init_group.add_argument("--preset-rubrics", default=None, help="Path to preset rubrics JSON")
    init_group.add_argument("--preset-strict", action="store_true", help="Require all prompts have preset")
    
    # === Content extraction ===
    extract_group = parser.add_argument_group("Content Extraction Options")
    extract_group.add_argument(
        "--extract-strategy",
        choices=["tag", "regex", "identity"],
        default="identity",
        help="Content extraction strategy (identity = no extraction)",
    )
    extract_group.add_argument("--extract-tag", default="content", help="Tag name for 'tag' strategy")
    extract_group.add_argument("--extract-pattern", default=None, help="Regex pattern for 'regex' strategy")
    extract_group.add_argument(
        "--extract-join-separator",
        default="\n\n",
        help="Separator for joining multiple extractions",
    )
    
    # === Logging ===
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    return parser


def build_runtime_config(args: Any) -> RuntimeConfig:
    """Build RuntimeConfig from parsed CLI arguments.
    
    This is the single point where CLI arguments are mapped to configuration objects.
    """
    api_key = os.getenv(args.api_key_env)
    
    return RuntimeConfig(
        backend=args.backend,
        llm=LLMBackendConfig(
            base_url=args.base_url,
            api_key=api_key,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
            default_model=args.model_default,
            initializer_model=args.model_initializer,
            proposer_model=args.model_proposer,
            verifier_model=args.model_verifier,
            judge_model=args.model_judge,
        ),
        search=SearchAlgorithmConfig(
            mode=args.mode,
            seed=args.seed,
            iterations=args.iterations,
            generations=args.generations,
            population_size=args.population_size,
            mutations_per_round=args.mutations_per_round,
            batch_size=args.batch_size,
        ),
        objective=ObjectiveFunctionConfig(
            tail_fraction=args.tail_fraction,
            lambda_var=args.lambda_var,
            mu_diverse=args.mu_diverse,
        ),
        initializer=InitializerStrategyConfig(
            strategy=args.initializer_strategy,
            preset_rubrics_path=args.preset_rubrics,
            strict=args.preset_strict,
        ),
        extraction=ContentExtractionConfig(
            strategy=args.extract_strategy,
            tag_name=args.extract_tag,
            pattern=args.extract_pattern,
            join_separator=args.extract_join_separator,
        ),
        verifier=VerifierConfig(noise=args.noise),
    )


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
        )
        logging.getLogger("autosr").setLevel(logging.INFO)


def compute_best_candidates(
    prompts: list[PromptExample],
    rubrics: dict[str, Rubric],
    factory: ComponentFactory,
    seed: int,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Compute best candidates and all candidate scores for each prompt.
    
    Returns:
        Tuple of (best_candidates, all_candidate_scores) where:
        - best_candidates: mapping from prompt_id to best candidate_id
        - all_candidate_scores: mapping from prompt_id to dict of candidate_id -> score
    """
    # Use the same extraction-aware verifier path as searcher creation.
    verifier = factory.create_verifier_with_extraction(prompts)
    evaluator = RubricEvaluator(verifier, base_seed=seed)
    
    best_candidates: dict[str, str] = {}
    all_candidate_scores: dict[str, dict[str, float]] = {}
    
    for item in prompts:
        rubric = rubrics.get(item.prompt_id)
        if rubric is None:
            continue
        scored = evaluator.evaluate_candidates(item, rubric)
        if not scored:
            continue
        best_candidates[item.prompt_id] = scored[0].candidate_id
        all_candidate_scores[item.prompt_id] = {
            ev.candidate_id: ev.score for ev in scored
        }
    
    return best_candidates, all_candidate_scores


def print_summary(
    scores: dict[str, float],
    mode: str,
    output_path: str,
    config: RuntimeConfig,
) -> None:
    """Print execution summary."""
    print(f"Mode: {mode}")
    print(f"Backend: {config.resolve_backend()}")
    
    resolved_backend = config.resolve_backend()
    if resolved_backend is BackendType.LLM:
        print("Models:")
        print(f"  - default: {config.llm.default_model}")
        print(f"  - initializer: {config.llm.get_model_for_role(LLMRole.INITIALIZER)}")
        print(f"  - proposer: {config.llm.get_model_for_role(LLMRole.PROPOSER)}")
        print(f"  - verifier: {config.llm.get_model_for_role(LLMRole.VERIFIER)}")
        print(f"  - judge: {config.llm.get_model_for_role(LLMRole.JUDGE)}")
    
    print("Best objective scores per prompt:")
    for prompt_id, score in sorted(scores.items()):
        print(f"  - {prompt_id}: {score:.4f}")
    
    print(f"Saved: {Path(output_path)}")


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        config = build_runtime_config(args)
    except ValueError as exc:
        parser.error(str(exc))
        return
    
    # Load dataset and create factory
    prompts = load_dataset(args.dataset)
    factory = ComponentFactory(config)
    
    # Create searcher and run search
    searcher = factory.create_searcher(prompts)
    result = searcher.search(prompts)
    
    # Compute best candidates
    best_candidates, candidate_scores = compute_best_candidates(
        prompts=prompts,
        rubrics=result.best_rubrics,
        factory=factory,
        seed=config.search.seed,
    )
    
    # Save results
    save_rubrics(
        args.output,
        result.best_rubrics,
        result.best_scores,
        best_candidates=best_candidates,
        candidate_scores=candidate_scores,
    )
    
    print_summary(result.best_scores, args.mode, args.output, config)


if __name__ == "__main__":
    main()
