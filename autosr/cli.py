from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from .evaluator import ObjectiveConfig, RubricEvaluator
from .io_utils import load_dataset, load_initial_rubrics, save_rubrics
from .llm_client import LLMClient
from .llm_components import (
    LLMPreferenceJudge,
    LLMRubricInitializer,
    LLMRubricProposer,
    LLMVerifier,
)
from .llm_config import LLMConfig, RoleModelConfig
from .content_extraction import create_verifier_with_extraction
from .mock_components import (
    HeuristicPreferenceJudge,
    HeuristicRubricInitializer,
    HeuristicVerifier,
    PresetRubricInitializer,
    RankPreferenceJudge,
    TemplateProposer,
)
from .models import PromptExample, Rubric
from .search import (
    EvolutionaryConfig,
    EvolutionaryRTDSearcher,
    IterativeConfig,
    IterativeRTDSearcher,
)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tail-focused rubric search runner")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON")
    parser.add_argument(
        "--mode",
        choices=["iterative", "evolutionary"],
        default="evolutionary",
        help="Search mode",
    )
    parser.add_argument("--output", required=True, help="Where to save best rubric JSON")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--noise", type=float, default=0.08, help="Verifier noise level")
    parser.add_argument(
        "--backend",
        choices=["auto", "mock", "llm"],
        default="auto",
        help="Backend selection: auto uses llm when api key exists, otherwise mock",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LLM API base URL")
    parser.add_argument(
        "--api-key-env",
        default="LLM_API_KEY",
        help="Environment variable name containing LLM API key",
    )
    parser.add_argument(
        "--model-default",
        default=DEFAULT_MODEL,
        help="Default LLM model for all roles",
    )
    parser.add_argument("--model-initializer", default=None, help="Optional model override for initializer")
    parser.add_argument("--model-proposer", default=None, help="Optional model override for proposer")
    parser.add_argument("--model-verifier", default=None, help="Optional model override for verifier")
    parser.add_argument("--model-judge", default=None, help="Optional model override for preference judge")
    parser.add_argument("--llm-timeout", type=float, default=30.0, help="LLM request timeout (seconds)")
    parser.add_argument("--llm-max-retries", type=int, default=2, help="LLM retry count")

    parser.add_argument(
        "--initializer-strategy",
        choices=["backend", "preset"],
        default="backend",
        help="Initial rubric strategy: backend uses default initializer for selected backend; preset loads from --preset-rubrics",
    )
    parser.add_argument(
        "--preset-rubrics",
        default=None,
        help="Path to preset initial rubric JSON (supports best_rubrics/rubrics/rubric formats)",
    )
    parser.add_argument(
        "--preset-strict",
        action="store_true",
        help="When using preset strategy, require every prompt_id to have a preset rubric",
    )

    parser.add_argument("--iterations", type=int, default=6, help="Iterative mode steps")
    parser.add_argument("--generations", type=int, default=12, help="Evolution generations")
    parser.add_argument("--population-size", type=int, default=8, help="Evolution population")
    parser.add_argument("--mutations-per-round", type=int, default=6, help="Mutations per generation")
    parser.add_argument("--batch-size", type=int, default=3, help="Prompts optimized per generation")

    parser.add_argument("--tail-fraction", type=float, default=0.25, help="Top fraction for tail metrics")
    parser.add_argument("--lambda-var", type=float, default=0.2, help="Penalty coefficient for tail variance")
    parser.add_argument("--mu-diverse", type=float, default=0.25, help="Bonus for cross-source tail alignment")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    # Content extraction options
    parser.add_argument(
        "--extract-strategy",
        choices=["tag", "regex", "identity"],
        default="identity",
        help="Content extraction strategy for verifier input (default: identity = no extraction)",
    )
    parser.add_argument(
        "--extract-tag",
        default="content",
        help="Tag name for 'tag' strategy (default: content)",
    )
    parser.add_argument(
        "--extract-pattern",
        default=None,
        help="Regex pattern for 'regex' strategy",
    )
    parser.add_argument(
        "--extract-join-separator",
        default="\n\n",
        help="Separator for joining multiple extractions (default: newline)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
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

    api_key = os.getenv(args.api_key_env)
    try:
        backend = resolve_backend(args.backend, api_key)
    except ValueError as exc:
        parser.error(str(exc))
        return

    role_models = build_role_model_config(args)
    objective = ObjectiveConfig(
        tail_fraction=args.tail_fraction,
        lambda_var=args.lambda_var,
        mu_diverse=args.mu_diverse,
    )
    prompts = load_dataset(args.dataset)
    proposer, verifier, judge, initializer = build_runtime_components(
        args=args,
        backend=backend,
        api_key=api_key,
        role_models=role_models,
        prompts=prompts,
    )

    if args.mode == "iterative":
        config = IterativeConfig(iterations=args.iterations, objective=objective, seed=args.seed)
        searcher = IterativeRTDSearcher(proposer, verifier, judge, initializer, config=config)
    else:
        config = EvolutionaryConfig(
            generations=args.generations,
            population_size=args.population_size,
            mutations_per_round=args.mutations_per_round,
            batch_size=args.batch_size,
            objective=objective,
            seed=args.seed,
        )
        searcher = EvolutionaryRTDSearcher(proposer, verifier, judge, initializer, config=config)

    result = searcher.search(prompts)
    best_candidates, candidate_scores = compute_best_candidates(
        prompts=prompts,
        rubrics=result.best_rubrics,
        verifier=verifier,
        seed=args.seed,
    )
    save_rubrics(
        args.output,
        result.best_rubrics,
        result.best_scores,
        best_candidates=best_candidates,
        candidate_scores=candidate_scores,
    )
    _print_summary(result.best_scores, args.mode, args.output, backend, role_models)


def resolve_backend(backend: str, api_key: str | None) -> str:
    if backend == "auto":
        return "llm" if api_key else "mock"
    if backend == "llm" and not api_key:
        raise ValueError(
            "LLM backend requires an API key. "
            "Set LLM_API_KEY or use --api-key-env to choose another variable."
        )
    return backend


def build_role_model_config(args: Any) -> RoleModelConfig:
    return RoleModelConfig(
        default=args.model_default,
        initializer=args.model_initializer,
        proposer=args.model_proposer,
        verifier=args.model_verifier,
        judge=args.model_judge,
    )


def _has_rank_for_all_candidates(prompts: list[PromptExample]) -> bool:
    """Check if all candidates across all prompts have metadata.rank defined."""
    for prompt in prompts:
        for candidate in prompt.candidates:
            if candidate.metadata.get("rank") is None:
                return False
    return True


def build_runtime_components(
    *,
    args: Any,
    backend: str,
    api_key: str | None,
    role_models: RoleModelConfig,
    prompts: list[PromptExample],
):
    if backend == "mock":
        # Auto-select judge based on dataset: if all candidates have rank, use RankPreferenceJudge
        if _has_rank_for_all_candidates(prompts):
            logging.getLogger(__name__).info("Detected metadata.rank in all candidates; using RankPreferenceJudge")
            judge: RankPreferenceJudge | HeuristicPreferenceJudge = RankPreferenceJudge()
        else:
            judge = HeuristicPreferenceJudge()
        proposer = TemplateProposer()
        verifier = HeuristicVerifier(noise=args.noise)
        initializer = HeuristicRubricInitializer()
    else:
        if api_key is None:
            raise ValueError("api_key is required for llm backend")
        llm_config_obj = LLMConfig(
            base_url=args.base_url,
            api_key=api_key,
            timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
            temperature=0.0,
        )
        requester = LLMClient(llm_config_obj)

        # Auto-select judge: if all candidates have rank, use RankPreferenceJudge instead of LLM
        if _has_rank_for_all_candidates(prompts):
            logging.getLogger(__name__).info("Detected metadata.rank in all candidates; using RankPreferenceJudge instead of LLMPreferenceJudge")
            judge = RankPreferenceJudge()
        else:
            judge = LLMPreferenceJudge(
                requester, model=role_models.for_role("judge"), max_retries=llm_config_obj.max_retries
            )

        proposer = LLMRubricProposer(
            requester, model=role_models.for_role("proposer"), max_retries=llm_config_obj.max_retries
        )
        verifier = LLMVerifier(requester, model=role_models.for_role("verifier"), max_retries=llm_config_obj.max_retries)
        initializer = LLMRubricInitializer(
            requester,
            model=role_models.for_role("initializer"),
            max_retries=llm_config_obj.max_retries,
        )

    if args.initializer_strategy == "preset":
        if not args.preset_rubrics:
            raise ValueError("--initializer-strategy preset requires --preset-rubrics")
        preset_rubrics = load_initial_rubrics(args.preset_rubrics)
        initializer = PresetRubricInitializer(
            preset_rubrics,
            fallback_initializer=initializer,
            strict=args.preset_strict,
        )

    # Apply content extraction strategy to verifier if requested
    if args.extract_strategy != "identity":
        extraction_kwargs = {
            "join_multiple": args.extract_join_separator,
        }
        if args.extract_strategy == "tag":
            extraction_kwargs["tag_name"] = args.extract_tag
        elif args.extract_strategy == "regex":
            if not args.extract_pattern:
                raise ValueError("--extract-strategy regex requires --extract-pattern")
            extraction_kwargs["pattern"] = args.extract_pattern

        verifier = create_verifier_with_extraction(
            verifier,
            args.extract_strategy,
            **extraction_kwargs,
        )

    return proposer, verifier, judge, initializer


def compute_best_candidates(
    *,
    prompts: list[PromptExample],
    rubrics: dict[str, Rubric],
    verifier: Any,
    seed: int,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Compute best candidates and all candidate scores for each prompt.
    
    Returns:
        Tuple of (best_candidates, all_candidate_scores) where:
        - best_candidates: mapping from prompt_id to best candidate_id
        - all_candidate_scores: mapping from prompt_id to dict of candidate_id -> score
    """
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


def _print_summary(
    scores: dict[str, float],
    mode: str,
    output_path: str,
    backend: str,
    models: RoleModelConfig,
) -> None:
    print(f"Mode: {mode}")
    print(f"Backend: {backend}")
    if backend == "llm":
        print("Models:")
        print(f"  - default: {models.default}")
        print(f"  - initializer: {models.for_role('initializer')}")
        print(f"  - proposer: {models.for_role('proposer')}")
        print(f"  - verifier: {models.for_role('verifier')}")
        print(f"  - judge: {models.for_role('judge')}")
    print("Best objective scores per prompt:")
    for prompt_id, score in sorted(scores.items()):
        print(f"  - {prompt_id}: {score:.4f}")
    print(f"Saved: {Path(output_path)}")


if __name__ == "__main__":
    main()
