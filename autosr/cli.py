from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from .evaluator import ObjectiveConfig, RubricEvaluator
from .io_utils import load_dataset, save_rubrics
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

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"


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
    parser.add_argument("--base-url", default=DEFAULT_OPENROUTER_BASE_URL, help="LLM API base URL")
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable name containing LLM API key",
    )
    parser.add_argument(
        "--model-default",
        default=DEFAULT_OPENROUTER_MODEL,
        help="Default LLM model for all roles",
    )
    parser.add_argument("--model-initializer", default=None, help="Optional model override for initializer")
    parser.add_argument("--model-proposer", default=None, help="Optional model override for proposer")
    parser.add_argument("--model-verifier", default=None, help="Optional model override for verifier")
    parser.add_argument("--model-judge", default=None, help="Optional model override for preference judge")
    parser.add_argument("--llm-timeout", type=float, default=30.0, help="LLM request timeout (seconds)")
    parser.add_argument("--llm-max-retries", type=int, default=2, help="LLM retry count")

    parser.add_argument("--iterations", type=int, default=6, help="Iterative mode steps")
    parser.add_argument("--generations", type=int, default=12, help="Evolution generations")
    parser.add_argument("--population-size", type=int, default=8, help="Evolution population")
    parser.add_argument("--mutations-per-round", type=int, default=6, help="Mutations per generation")
    parser.add_argument("--batch-size", type=int, default=3, help="Prompts optimized per generation")

    parser.add_argument("--tail-fraction", type=float, default=0.25, help="Top fraction for tail metrics")
    parser.add_argument("--lambda-var", type=float, default=0.2, help="Penalty coefficient for tail variance")
    parser.add_argument("--mu-diverse", type=float, default=0.25, help="Bonus for cross-source tail alignment")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
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
    best_candidates = compute_best_candidates(
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
    )
    _print_summary(result.best_scores, args.mode, args.output, backend, role_models)


def resolve_backend(backend: str, api_key: str | None) -> str:
    if backend == "auto":
        return "llm" if api_key else "mock"
    if backend == "llm" and not api_key:
        raise ValueError(
            "LLM backend requires an API key. "
            "Set OPENROUTER_API_KEY or use --api-key-env to choose another variable."
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
        return (
            TemplateProposer(),
            HeuristicVerifier(noise=args.noise),
            judge,
            HeuristicRubricInitializer(),
        )

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
        judge: LLMPreferenceJudge | RankPreferenceJudge = RankPreferenceJudge()
    else:
        judge = LLMPreferenceJudge(
            requester, model=role_models.for_role("judge"), max_retries=llm_config_obj.max_retries
        )
    
    return (
        LLMRubricProposer(
            requester, model=role_models.for_role("proposer"), max_retries=llm_config_obj.max_retries
        ),
        LLMVerifier(requester, model=role_models.for_role("verifier"), max_retries=llm_config_obj.max_retries),
        judge,
        LLMRubricInitializer(
            requester,
            model=role_models.for_role("initializer"),
            max_retries=llm_config_obj.max_retries,
        ),
    )


def compute_best_candidates(
    *,
    prompts: list[PromptExample],
    rubrics: dict[str, Rubric],
    verifier: Any,
    seed: int,
) -> dict[str, str]:
    evaluator = RubricEvaluator(verifier, base_seed=seed)
    best_candidates: dict[str, str] = {}
    for item in prompts:
        rubric = rubrics.get(item.prompt_id)
        if rubric is None:
            continue
        scored = evaluator.evaluate_candidates(item, rubric)
        if not scored:
            continue
        best_candidates[item.prompt_id] = scored[0].candidate_id
    return best_candidates


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
