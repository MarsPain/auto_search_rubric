from __future__ import annotations

import json
import random
from typing import Any
import unittest

from autosr.exceptions import LLMCallError, LLMFatalCallError, LLMParseError
from autosr.llm_components import (
    LLMPreferenceJudge,
    LLMRubricInitializer,
    LLMRubricProposer,
    LLMVerifier,
)
from autosr.models import Criterion, PromptExample, ResponseCandidate, Rubric
from autosr.types import MutationMode


class _StubRequester:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.calls = 0
        self.requests: list[dict[str, str]] = []

    def request_json(self, *, model: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.requests.append(
            {
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


class _RaisingRequester:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def request_json(self, *, model: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        del model, system_prompt, user_prompt
        self.calls += 1
        raise self.error


def _build_item() -> PromptExample:
    return PromptExample(
        prompt_id="p1",
        prompt="Write a concise plan",
        candidates=[
            ResponseCandidate(candidate_id="a", text="alpha", source="strong"),
            ResponseCandidate(candidate_id="b", text="beta", source="base"),
        ],
    )


def _build_rubric() -> Rubric:
    return Rubric(
        rubric_id="r_base",
        criteria=[
            Criterion("c1", "Task fit", weight=0.6),
            Criterion("c2", "Specificity", weight=0.4),
        ],
    )


class TestLLMComponents(unittest.TestCase):
    def test_initializer_parses_rubric(self) -> None:
        requester = _StubRequester(
            [
                {
                    "rubric_id": "rubric_from_llm",
                    "criteria": [
                        {
                            "criterion_id": "task_fit",
                            "text": "Matches task requirements",
                            "weight": 1.0,
                            "criterion_type": "factual",
                        }
                    ],
                    "grading_protocol": {"num_votes": 1, "allow_na": True, "vote_method": "majority"},
                }
            ]
        )
        component = LLMRubricInitializer(requester, model="openai/gpt-4o-mini", max_retries=1)
        rubric = component.initialize(_build_item(), rng=random.Random(1))

        self.assertEqual(rubric.rubric_id, "rubric_from_llm")
        self.assertEqual(len(rubric.criteria), 1)
        self.assertEqual(rubric.grading_protocol.num_votes, 1)

    def test_proposer_parses_mutated_rubric(self) -> None:
        requester = _StubRequester(
            [
                {
                    "rubric_id": "rubric_m1",
                    "criteria": [
                        {"criterion_id": "c1", "text": "Task fit", "weight": 0.7},
                        {"criterion_id": "c2", "text": "Specificity", "weight": 0.3},
                    ],
                }
            ]
        )
        component = LLMRubricProposer(requester, model="openai/gpt-4o-mini", max_retries=1)
        mutated = component.propose(
            _build_item().prompt,
            _build_item().candidates[0],
            _build_item().candidates[1],
            _build_rubric(),
            mode=MutationMode.RAISE_BAR,
            rng=random.Random(1),
        )
        self.assertEqual(mutated.rubric_id, "rubric_m1")
        self.assertEqual(len(mutated.criteria), 2)

    def test_verifier_maps_grade_types(self) -> None:
        requester = _StubRequester([{"grades": {"c1": "1", "c2": "n/a"}}])
        component = LLMVerifier(requester, model="openai/gpt-4o-mini", max_retries=1)
        grades = component.grade(
            prompt="prompt",
            candidate=ResponseCandidate(candidate_id="a", text="hello"),
            rubric=_build_rubric(),
            seed=42,
        )
        self.assertEqual(grades, {"c1": 1, "c2": None})

    def test_verifier_supports_continuous_scores(self) -> None:
        requester = _StubRequester([{"grades": {"c1": 4.5, "c2": "0.2"}}])
        component = LLMVerifier(requester, model="openai/gpt-4o-mini", max_retries=1)
        grades = component.grade(
            prompt="prompt",
            candidate=ResponseCandidate(candidate_id="a", text="hello"),
            rubric=_build_rubric(),
            seed=42,
        )
        self.assertEqual(grades, {"c1": 4.5, "c2": 0.2})

    def test_judge_invalid_output_retries_then_fails(self) -> None:
        requester = _StubRequester([{"preference": 2}, {"preference": 9}])
        component = LLMPreferenceJudge(requester, model="openai/gpt-4o-mini", max_retries=1)

        with self.assertRaises(LLMParseError):
            component.compare(
                prompt="prompt",
                left=ResponseCandidate(candidate_id="a", text="left"),
                right=ResponseCandidate(candidate_id="b", text="right"),
            )
        self.assertEqual(requester.calls, 2)

    def test_initializer_repairs_missing_criterion_text(self) -> None:
        requester = _StubRequester(
            [
                {
                    "rubric_id": "rubric_bad",
                    "criteria": [
                        {"criterion_id": "email_structure_conciseness", "weight": 1.0},
                    ],
                }
            ]
        )
        component = LLMRubricInitializer(requester, model="openai/gpt-4o-mini", max_retries=0)

        rubric = component.initialize(_build_item(), rng=random.Random(1))

        self.assertEqual(rubric.rubric_id, "rubric_bad")
        self.assertEqual(rubric.criteria[0].criterion_id, "email_structure_conciseness")
        self.assertNotEqual(rubric.criteria[0].text.strip(), "")

    def test_initializer_prompt_includes_required_criterion_fields(self) -> None:
        requester = _StubRequester(
            [
                {
                    "rubric_id": "r1",
                    "criteria": [{"criterion_id": "c1", "text": "Task fit", "weight": 1.0}],
                }
            ]
        )
        component = LLMRubricInitializer(requester, model="openai/gpt-4o-mini", max_retries=0)

        component.initialize(_build_item(), rng=random.Random(1))

        payload = json.loads(requester.requests[0]["user_prompt"])
        constraints = payload["constraints"]
        self.assertEqual(constraints["criterion_required_fields"], ["criterion_id", "text", "weight"])
        self.assertTrue(constraints["return_full_rubric"])

    def test_proposer_prompt_requires_full_rubric_not_patch(self) -> None:
        requester = _StubRequester(
            [
                {
                    "rubric_id": "r2",
                    "criteria": [{"criterion_id": "c1", "text": "Task fit", "weight": 1.0}],
                }
            ]
        )
        component = LLMRubricProposer(requester, model="openai/gpt-4o-mini", max_retries=0)

        component.propose(
            _build_item().prompt,
            _build_item().candidates[0],
            _build_item().candidates[1],
            _build_rubric(),
            mode=MutationMode.RAISE_BAR,
            rng=random.Random(1),
        )

        payload = json.loads(requester.requests[0]["user_prompt"])
        output_requirements = payload["output_requirements"]
        self.assertEqual(
            output_requirements["criterion_required_fields"],
            ["criterion_id", "text", "weight"],
        )
        self.assertTrue(output_requirements["must_return_full_rubric"])

    def test_proposer_prompt_includes_mode_requirements_contract(self) -> None:
        requester = _StubRequester(
            [
                {
                    "rubric_id": "r3",
                    "criteria": [{"criterion_id": "c1", "text": "Task fit", "weight": 1.0}],
                }
            ]
        )
        component = LLMRubricProposer(requester, model="openai/gpt-4o-mini", max_retries=0)

        component.propose(
            _build_item().prompt,
            _build_item().candidates[0],
            _build_item().candidates[1],
            _build_rubric(),
            mode=MutationMode.DECOMPOSE,
            rng=random.Random(1),
        )

        payload = json.loads(requester.requests[0]["user_prompt"])
        output_requirements = payload["output_requirements"]
        self.assertIn("mode_requirements", output_requirements)
        mode_requirements = output_requirements["mode_requirements"]
        expected_modes = {
            "raise_bar",
            "decompose",
            "factual_focus",
            "anti_fluff",
            "counterexample_trigger",
            "weight_perturb",
        }
        self.assertEqual(set(mode_requirements.keys()), expected_modes)
        self.assertTrue(mode_requirements["decompose"]["must_split_one_criterion"])
        self.assertTrue(mode_requirements["weight_perturb"]["must_preserve_criterion_text"])

    def test_initializer_fail_soft_falls_back_to_heuristic(self) -> None:
        requester = _RaisingRequester(LLMCallError("timeout"))
        component = LLMRubricInitializer(
            requester,
            model="openai/gpt-4o-mini",
            max_retries=0,
            fail_soft=True,
        )

        rubric = component.initialize(_build_item(), rng=random.Random(1))

        self.assertEqual(rubric.metadata.get("origin"), "heuristic_initializer")
        self.assertGreater(len(rubric.criteria), 0)
        self.assertEqual(requester.calls, 1)

    def test_proposer_fail_soft_returns_current_rubric(self) -> None:
        requester = _RaisingRequester(LLMCallError("timeout"))
        component = LLMRubricProposer(
            requester,
            model="openai/gpt-4o-mini",
            max_retries=0,
            fail_soft=True,
        )
        current = _build_rubric()

        mutated = component.propose(
            _build_item().prompt,
            _build_item().candidates[0],
            _build_item().candidates[1],
            current,
            mode=MutationMode.RAISE_BAR,
            rng=random.Random(1),
        )

        self.assertIs(mutated, current)
        self.assertEqual(requester.calls, 1)

    def test_verifier_fail_soft_returns_na_grades(self) -> None:
        requester = _RaisingRequester(LLMCallError("timeout"))
        component = LLMVerifier(
            requester,
            model="openai/gpt-4o-mini",
            max_retries=0,
            fail_soft=True,
        )
        rubric = _build_rubric()

        grades = component.grade(
            prompt="prompt",
            candidate=ResponseCandidate(candidate_id="a", text="hello"),
            rubric=rubric,
            seed=42,
        )

        self.assertEqual(grades, {"c1": None, "c2": None})
        self.assertEqual(requester.calls, 1)

    def test_judge_fail_soft_returns_tie(self) -> None:
        requester = _RaisingRequester(LLMCallError("timeout"))
        component = LLMPreferenceJudge(
            requester,
            model="openai/gpt-4o-mini",
            max_retries=0,
            fail_soft=True,
        )

        pref = component.compare(
            prompt="prompt",
            left=ResponseCandidate(candidate_id="a", text="left"),
            right=ResponseCandidate(candidate_id="b", text="right"),
        )

        self.assertEqual(pref, 0)
        self.assertEqual(requester.calls, 1)

    def test_fail_soft_does_not_swallow_fatal_call_error(self) -> None:
        requester = _RaisingRequester(LLMFatalCallError("invalid api key"))
        component = LLMRubricProposer(
            requester,
            model="openai/gpt-4o-mini",
            max_retries=0,
            fail_soft=True,
        )

        with self.assertRaises(LLMFatalCallError):
            component.propose(
                _build_item().prompt,
                _build_item().candidates[0],
                _build_item().candidates[1],
                _build_rubric(),
                mode=MutationMode.RAISE_BAR,
                rng=random.Random(1),
            )
        self.assertEqual(requester.calls, 1)


if __name__ == "__main__":
    unittest.main()
