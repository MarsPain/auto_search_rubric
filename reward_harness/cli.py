"""CLI entry point for tail-focused rubric search.

This module provides a minimal CLI that delegates all component creation
to ComponentFactory and all configuration to RuntimeConfig.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys
from typing import Any, Callable

from .config import (
    CandidateTextExtractionConfig,
    ContentExtractionConfig,
    InitializerStrategyConfig,
    LLMBackendConfig,
    ObjectiveFunctionConfig,
    RuntimeConfig,
    SearchAlgorithmConfig,
    VerifierConfig,
)
from .search import SearchResult
from .types import (
    AdaptiveMutationSchedule,
    BackendType,
    EvolutionIterationScope,
    LLMRole,
    SelectionStrategy,
)
from .evaluator import RubricEvaluator
from .factory import ComponentFactory
from .run_records.use_cases import build_reproducible_script, build_run_manifest
from .io_utils import (
    load_dataset,
    save_run_record_files,
    save_rubrics,
)
from .data_models import PromptExample, Rubric
from .harness import SearchSession, StateManager

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "stepfun/step-3.5-flash:free"
SELECTION_STRATEGY_CHOICES = [strategy.value for strategy in SelectionStrategy]
ADAPTIVE_MUTATION_CHOICES = [schedule.value for schedule in AdaptiveMutationSchedule]
EVOLUTION_ITERATION_SCOPE_CHOICES = [scope.value for scope in EvolutionIterationScope]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with logically grouped arguments."""
    parser = argparse.ArgumentParser(
        description="Tail-focused rubric search runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # === Core arguments ===
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON")
    parser.add_argument(
        "--output", required=True, help="Where to save best rubric JSON"
    )
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
    llm_group.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="LLM API base URL"
    )
    llm_group.add_argument(
        "--model-default", default=DEFAULT_MODEL, help="Default LLM model"
    )
    llm_group.add_argument(
        "--model-initializer", default=None, help="Model for initializer"
    )
    llm_group.add_argument("--model-proposer", default=None, help="Model for proposer")
    llm_group.add_argument("--model-verifier", default=None, help="Model for verifier")
    llm_group.add_argument("--model-judge", default=None, help="Model for judge")
    llm_group.add_argument(
        "--llm-timeout", type=float, default=30.0, help="Request timeout (s)"
    )
    llm_group.add_argument("--llm-max-retries", type=int, default=2, help="Retry count")
    llm_group.add_argument(
        "--llm-retry-backoff-base",
        type=float,
        default=0.5,
        help="Base seconds for exponential retry backoff",
    )
    llm_group.add_argument(
        "--llm-retry-backoff-max",
        type=float,
        default=8.0,
        help="Maximum seconds for exponential retry backoff",
    )
    llm_group.add_argument(
        "--llm-retry-jitter",
        type=float,
        default=0.2,
        help="Relative jitter ratio applied to retry delay (0 disables jitter)",
    )
    llm_group.add_argument(
        "--llm-fail-soft",
        action="store_true",
        help="Keep search running by applying role-specific fallbacks when an LLM call exhausts retries",
    )
    llm_group.add_argument(
        "--prompt-language",
        default="zh",
        help="Language for LLM prompt templates (loads from prompts/<lang>/, fallback prompts/; default: constants)",
    )

    # === Search algorithm parameters ===
    search_group = parser.add_argument_group("Search Algorithm Options")
    search_group.add_argument("--seed", type=int, default=7, help="Random seed")
    search_group.add_argument(
        "--iterations", type=int, default=6, help="[Iterative] steps"
    )
    search_group.add_argument(
        "--generations", type=int, default=12, help="[Evolutionary] generations"
    )
    search_group.add_argument(
        "--population-size", type=int, default=8, help="[Evolutionary] population"
    )
    search_group.add_argument(
        "--mutations-per-round", type=int, default=6, help="Mutations per round"
    )
    search_group.add_argument(
        "--mutation-parent-count",
        type=int,
        default=3,
        help="Number of parent rubrics to mutate per generation",
    )
    search_group.add_argument(
        "--batch-size", type=int, default=3, help="Prompts per generation"
    )
    search_group.add_argument(
        "--evolution-iteration-scope",
        choices=EVOLUTION_ITERATION_SCOPE_CHOICES,
        default=EvolutionIterationScope.GLOBAL_BATCH.value,
        help="Evolution scheduling scope: global_batch selects hard prompts per generation, prompt_local evolves each prompt independently",
    )
    search_group.add_argument(
        "--stop-when-distinguished",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In prompt_local scope, stop a prompt early once top candidates are separated by margin",
    )
    search_group.add_argument(
        "--distinguish-margin",
        type=float,
        default=None,
        help="Override top-margin threshold for early stop in prompt_local scope (default: objective tie tolerance)",
    )

    # === Selection Strategy Options ===
    selection_group = parser.add_argument_group(
        "Selection Strategy Options (Evolutionary)"
    )
    selection_group.add_argument(
        "--selection-strategy",
        choices=SELECTION_STRATEGY_CHOICES,
        default=SelectionStrategy.RANK.value,
        help="Parent selection strategy (default: rank)",
    )
    selection_group.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="Tournament size for tournament selection",
    )
    selection_group.add_argument(
        "--tournament-p",
        type=float,
        default=0.8,
        help="Probability of selecting best from tournament",
    )
    selection_group.add_argument(
        "--top-k-ratio",
        type=float,
        default=0.3,
        help="Ratio of population as elite pool for top_k selection",
    )
    selection_group.add_argument(
        "--diversity-weight",
        type=float,
        default=0.3,
        help="Weight for diversity in selection score (0-1)",
    )

    # === Adaptive Mutation Options ===
    mutation_group = parser.add_argument_group(
        "Adaptive Mutation Options (Evolutionary)"
    )
    mutation_group.add_argument(
        "--adaptive-mutation",
        choices=ADAPTIVE_MUTATION_CHOICES,
        default=AdaptiveMutationSchedule.FIXED.value,
        help="Adaptive mutation schedule (default: fixed)",
    )
    mutation_group.add_argument(
        "--mutation-window-size",
        type=int,
        default=10,
        help="History window size for tracking mutation success",
    )
    mutation_group.add_argument(
        "--min-mutation-weight",
        type=float,
        default=0.1,
        help="Minimum weight for any mutation mode",
    )
    mutation_group.add_argument(
        "--exploration-phase-ratio",
        type=float,
        default=0.3,
        help="Ratio of generations for exploration phase",
    )
    mutation_group.add_argument(
        "--diversity-threshold",
        type=float,
        default=0.05,
        help="Diversity threshold for triggering diversity-boosting mutations",
    )

    # === Objective function parameters ===
    objective_group = parser.add_argument_group("Objective Function Options")
    objective_group.add_argument(
        "--tail-fraction", type=float, default=0.25, help="Top fraction for tail"
    )
    objective_group.add_argument(
        "--lambda-var", type=float, default=0.2, help="Variance penalty coefficient"
    )
    objective_group.add_argument(
        "--mu-diverse", type=float, default=0.25, help="Diversity bonus coefficient"
    )
    objective_group.add_argument(
        "--pair-confidence-prior",
        type=float,
        default=8.0,
        help="Pseudo-count prior for shrinking pair-based accuracies toward 0.5",
    )

    # === Verifier options ===
    verifier_group = parser.add_argument_group("Verifier Options")
    verifier_group.add_argument(
        "--noise", type=float, default=0.08, help="Verifier noise level (mock)"
    )

    # === Initializer strategy ===
    init_group = parser.add_argument_group("Initializer Strategy Options")
    init_group.add_argument(
        "--initializer-strategy",
        choices=["backend", "preset"],
        default="backend",
        help="Strategy for initial rubrics",
    )
    init_group.add_argument(
        "--preset-rubrics", default=None, help="Path to preset rubrics JSON"
    )
    init_group.add_argument(
        "--preset-strict", action="store_true", help="Require all prompts have preset"
    )

    # === Content extraction ===
    extract_group = parser.add_argument_group("Content Extraction Options")
    extract_group.add_argument(
        "--extract-strategy",
        choices=["tag", "regex", "identity"],
        default="identity",
        help="Content extraction strategy (identity = no extraction)",
    )
    extract_group.add_argument(
        "--extract-tag", default="content", help="Tag name for 'tag' strategy"
    )
    extract_group.add_argument(
        "--extract-pattern", default=None, help="Regex pattern for 'regex' strategy"
    )
    extract_group.add_argument(
        "--extract-join-separator",
        default="\n\n",
        help="Separator for joining multiple extractions",
    )

    # === Candidate text extraction ===
    candidate_extract_group = parser.add_argument_group(
        "Candidate Text Extraction Options"
    )
    candidate_extract_group.add_argument(
        "--candidate-extract-strategy",
        choices=["answer", "identity"],
        default="answer",
        help="Candidate text extraction strategy (answer = prefer <answer>, else strip <think>)",
    )
    candidate_extract_group.add_argument(
        "--candidate-extract-join-separator",
        default="\n\n",
        help="Separator when multiple <answer> segments are present",
    )

    # === Checkpoint/Resume options ===
    checkpoint_group = parser.add_argument_group("Checkpoint and Resume Options")
    checkpoint_group.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Directory for storing checkpoints (default: ./checkpoints)",
    )
    checkpoint_group.add_argument(
        "--checkpoint-every-generation",
        action="store_true",
        help="Save checkpoint after each generation (evolutionary mode only)",
    )
    checkpoint_group.add_argument(
        "--checkpoint-interval-seconds",
        type=float,
        default=None,
        help="Minimum seconds between checkpoints",
    )
    checkpoint_group.add_argument(
        "--resume-from",
        default=None,
        help="Resume from session_id or checkpoint path",
    )

    # === Logging ===
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

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
            retry_backoff_base=getattr(args, "llm_retry_backoff_base", 0.5),
            retry_backoff_max=getattr(args, "llm_retry_backoff_max", 8.0),
            retry_jitter=getattr(args, "llm_retry_jitter", 0.2),
            fail_soft=getattr(args, "llm_fail_soft", False),
            default_model=args.model_default,
            initializer_model=args.model_initializer,
            proposer_model=args.model_proposer,
            verifier_model=args.model_verifier,
            judge_model=args.model_judge,
            prompt_language=getattr(args, "prompt_language", None),
        ),
        search=SearchAlgorithmConfig(
            mode=args.mode,
            seed=args.seed,
            iterations=args.iterations,
            generations=args.generations,
            population_size=args.population_size,
            mutations_per_round=args.mutations_per_round,
            mutation_parent_count=getattr(args, "mutation_parent_count", 3),
            batch_size=args.batch_size,
            iteration_scope=args.evolution_iteration_scope,
            stop_when_distinguished=args.stop_when_distinguished,
            distinguish_margin=args.distinguish_margin,
            selection_strategy=args.selection_strategy,
            tournament_size=args.tournament_size,
            tournament_p=args.tournament_p,
            top_k_ratio=args.top_k_ratio,
            diversity_weight=args.diversity_weight,
            adaptive_mutation=args.adaptive_mutation,
            mutation_window_size=args.mutation_window_size,
            min_mutation_weight=args.min_mutation_weight,
            exploration_phase_ratio=args.exploration_phase_ratio,
            diversity_threshold=args.diversity_threshold,
        ),
        objective=ObjectiveFunctionConfig(
            tail_fraction=args.tail_fraction,
            lambda_var=args.lambda_var,
            mu_diverse=args.mu_diverse,
            pair_confidence_prior=args.pair_confidence_prior,
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
        candidate_extraction=CandidateTextExtractionConfig(
            strategy=args.candidate_extract_strategy,
            join_separator=args.candidate_extract_join_separator,
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
        logging.getLogger("reward_harness").setLevel(logging.INFO)


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
    *,
    run_manifest_path: Path | None = None,
    reproducible_script_path: Path | None = None,
    session_info: dict[str, Any] | None = None,
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

    # Print session info if using harness
    if session_info:
        print("Session:")
        print(f"  - session_id: {session_info.get('session_id', 'unknown')}")
        if session_info.get("is_resumed"):
            print(f"  - resumed_from: {session_info.get('resume_source', 'unknown')}")
        if session_info.get("checkpoint_enabled"):
            print("  - checkpoint_enabled: true")

    print("Best objective scores per prompt:")
    for prompt_id, score in sorted(scores.items()):
        print(f"  - {prompt_id}: {score:.4f}")

    print(f"Saved: {Path(output_path)}")
    if run_manifest_path is not None:
        print(f"Run manifest: {run_manifest_path}")
    if reproducible_script_path is not None:
        print(f"Repro script: {reproducible_script_path}")


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    raw_cli_args = sys.argv[1:]
    args = parser.parse_args(raw_cli_args)

    setup_logging(args.verbose)

    try:
        config = build_runtime_config(args)
    except ValueError as exc:
        parser.error(str(exc))
        return

    # Load dataset and create factory
    prompts = load_dataset(args.dataset)
    factory = ComponentFactory(config)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_manifest = build_run_manifest(
        args=args,
        config=config,
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        raw_cli_args=raw_cli_args,
        run_id=run_id,
    )
    reproducible_script = build_reproducible_script(run_manifest)

    # Determine whether to use harness session with checkpointing
    use_harness = (
        args.checkpoint_every_generation
        or args.resume_from
        or args.checkpoint_interval_seconds is not None
    )

    result: SearchResult | None = None
    session_info: dict[str, Any] = {}

    if use_harness and config.search.is_evolutionary():
        # Use harness session with checkpointing support
        state_manager = StateManager(base_dir=args.checkpoint_dir)

        if args.resume_from:
            # Resume from checkpoint
            dataset_path = Path(args.dataset)
            session = SearchSession.resume(
                resume_from=args.resume_from,
                config=config,
                factory=factory,
                state_manager=state_manager,
                prompts=prompts,
                dataset_path=dataset_path,
                checkpoint_every_generation=args.checkpoint_every_generation,
                checkpoint_interval_seconds=args.checkpoint_interval_seconds,
            )
        else:
            # Create new session
            session = SearchSession.create(
                prompts=prompts,
                config=config,
                factory=factory,
                session_id=run_id.rstrip("Z").replace("T", "_"),
                state_manager=state_manager,
                checkpoint_every_generation=args.checkpoint_every_generation,
                checkpoint_interval_seconds=args.checkpoint_interval_seconds,
                dataset_path=Path(args.dataset),
            )

        result = session.run_to_completion()
        session_info = session.get_session_info()
    else:
        # Use traditional searcher path
        checkpoint_callback: (
            Callable[
                [dict[str, Rubric], dict[str, float], dict[str, list[float]]], None
            ]
            | None
        ) = None
        if (
            config.search.is_evolutionary()
            and config.search.iteration_scope is EvolutionIterationScope.PROMPT_LOCAL
        ):

            def _checkpoint_callback(
                best_rubrics: dict[str, Rubric],
                best_scores: dict[str, float],
                _history: dict[str, list[float]],
            ) -> None:
                save_rubrics(
                    args.output,
                    best_rubrics,
                    best_scores,
                    run_manifest=run_manifest,
                )

            checkpoint_callback = _checkpoint_callback

        # Create searcher and run search
        searcher = factory.create_searcher(
            prompts, checkpoint_callback=checkpoint_callback
        )
        result = searcher.search(prompts)

    # Compute best candidates
    best_candidates, candidate_scores = compute_best_candidates(
        prompts=prompts,
        rubrics=result.best_rubrics,
        factory=factory,
        seed=config.search.seed,
    )

    if session_info:
        run_manifest["harness"] = {
            "session_id": session_info.get("session_id"),
            "is_resumed": session_info.get("is_resumed"),
            "resume_source": session_info.get("resume_source"),
            "resume_semantics": session_info.get("resume_semantics"),
            "checkpoint_every_generation": session_info.get(
                "checkpoint_every_generation"
            ),
            "checkpoint_interval_seconds": session_info.get(
                "checkpoint_interval_seconds"
            ),
        }

    # Save results
    save_rubrics(
        args.output,
        result.best_rubrics,
        result.best_scores,
        best_candidates=best_candidates,
        candidate_scores=candidate_scores,
        run_manifest=run_manifest,
        search_diagnostics=result.diagnostics,
    )
    run_manifest_path, reproducible_script_path = save_run_record_files(
        args.output,
        run_manifest=run_manifest,
        reproducible_script=reproducible_script,
    )

    print_summary(
        result.best_scores,
        args.mode,
        args.output,
        config,
        run_manifest_path=run_manifest_path,
        reproducible_script_path=reproducible_script_path,
        session_info=session_info,
    )


if __name__ == "__main__":
    main()
