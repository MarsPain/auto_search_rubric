from __future__ import annotations

import unittest

from autosr.mock_components import (
    HeuristicPreferenceJudge,
    HeuristicRubricInitializer,
    HeuristicVerifier,
    TemplateProposer,
)
from autosr.models import PromptExample, ResponseCandidate
from autosr.search import EvolutionaryConfig, EvolutionaryRTDSearcher
from autosr.types import MutationMode


def _build_prompt(prompt_id: str) -> PromptExample:
    return PromptExample(
        prompt_id=prompt_id,
        prompt="Assess the response quality.",
        candidates=[
            ResponseCandidate(candidate_id="a", text="First response with concrete example.", source="s1"),
            ResponseCandidate(candidate_id="b", text="Second response with details and steps.", source="s2"),
            ResponseCandidate(candidate_id="c", text="Third response with concise structure.", source="s3"),
        ],
    )


class StubMutationScheduler:
    def __init__(self) -> None:
        self.select_calls: list[float | None] = []
        self.record_calls: list[tuple[MutationMode, bool, float]] = []
        self.next_generation_calls = 0

    def select_mode(self, diversity_score: float | None = None) -> MutationMode:
        self.select_calls.append(diversity_score)
        return MutationMode.RAISE_BAR

    def record_outcome(
        self,
        mode: MutationMode,
        was_successful: bool,
        score_improvement: float,
    ) -> None:
        self.record_calls.append((mode, was_successful, score_improvement))

    def next_generation(self) -> None:
        self.next_generation_calls += 1

    def get_diagnostics(self) -> dict[str, str]:
        return {"source": "stub-scheduler"}


class CountingDiversityMetric:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def compute(self, rubrics: list, *, rng=None) -> float:  # noqa: ANN001
        self.calls += 1
        return self.value


class TestEvolutionaryDecoupling(unittest.TestCase):
    def test_search_uses_injected_scheduler_and_single_pass_diversity_metric(self) -> None:
        prompts = [_build_prompt("p1"), _build_prompt("p2")]
        config = EvolutionaryConfig(
            generations=2,
            population_size=4,
            mutations_per_round=2,
            batch_size=2,
            stagnation_generations=8,
            seed=7,
        )
        scheduler = StubMutationScheduler()
        diversity_metric = CountingDiversityMetric(value=0.42)

        searcher = EvolutionaryRTDSearcher(
            proposer=TemplateProposer(),
            verifier=HeuristicVerifier(noise=0.0),
            judge=HeuristicPreferenceJudge(),
            initializer=HeuristicRubricInitializer(),
            config=config,
            mutation_scheduler=scheduler,
            diversity_metric=diversity_metric,
        )

        result = searcher.search(prompts)

        self.assertEqual(diversity_metric.calls, config.generations * len(prompts))
        self.assertEqual(scheduler.next_generation_calls, config.generations)
        self.assertGreater(len(scheduler.select_calls), 0)
        self.assertTrue(all(score == 0.42 for score in scheduler.select_calls))
        self.assertGreater(len(scheduler.record_calls), 0)
        self.assertEqual(result.diagnostics["mutation_diagnostics"], {"source": "stub-scheduler"})
        self.assertAlmostEqual(result.diagnostics["avg_diversity"], 0.42)


if __name__ == "__main__":
    unittest.main()
