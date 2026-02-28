from __future__ import annotations

import json
import random
import unittest
from typing import Any

from autosr.exceptions import LLMParseError
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


if __name__ == "__main__":
    unittest.main()
