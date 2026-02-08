from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import random
import re

from .models import Criterion, GradingProtocol, PromptExample, ResponseCandidate, Rubric


class HeuristicRubricInitializer:
    """
    Demo initializer. In production, replace with an LLM-based proposer.
    """

    def initialize(self, item: PromptExample, *, rng: random.Random) -> Rubric:
        keywords = _top_keywords(item.prompt, max_keywords=4)
        criteria = [
            Criterion(
                criterion_id="task_fit",
                text="Response directly addresses the user task and constraints.",
                weight=0.35,
                criterion_type="factual",
                check_points=keywords,
            ),
            Criterion(
                criterion_id="structure",
                text="Response has clear structure with useful sections and transitions.",
                weight=0.25,
                criterion_type="format",
                check_points=["first", "second", "1.", "2."],
            ),
            Criterion(
                criterion_id="specificity",
                text="Response includes concrete details instead of generic statements.",
                weight=0.25,
                criterion_type="reasoning",
                check_points=["because", "example", "detail", "step"],
            ),
            Criterion(
                criterion_id="safety_style",
                text="Response avoids harmful or low-signal filler content.",
                weight=0.15,
                criterion_type="safety",
                negative_examples=["as an ai language model", "cannot provide"],
            ),
        ]
        return Rubric(
            rubric_id=f"{item.prompt_id}_r0",
            criteria=criteria,
            grading_protocol=GradingProtocol(num_votes=3, allow_na=True, vote_method="majority"),
            metadata={"origin": "heuristic_initializer"},
        )


class HeuristicVerifier:
    """
    Demo verifier with deterministic noise to emulate stochastic LLM grading.
    """

    def __init__(self, *, noise: float = 0.08) -> None:
        self.noise = noise

    def grade(
        self,
        prompt: str,
        candidate: ResponseCandidate,
        rubric: Rubric,
        *,
        seed: int,
    ) -> dict[str, int | None]:
        text = candidate.text.lower()
        out: dict[str, int | None] = {}
        rng = random.Random(seed)
        for criterion in rubric.criteria:
            value = self._grade_criterion(text, criterion)
            # Add low-rate deterministic noise to mimic variance in graders.
            if rng.random() < self.noise:
                value = 1 - value
            out[criterion.criterion_id] = value
        return out

    def _grade_criterion(self, text: str, criterion: Criterion) -> int:
        if criterion.negative_examples:
            for pattern in criterion.negative_examples:
                if pattern.lower() in text:
                    return 0

        if criterion.check_points:
            hits = sum(1 for cp in criterion.check_points if cp.lower() in text)
            ratio = hits / max(1, len(criterion.check_points))
            return 1 if ratio >= 0.5 else 0

        # Fallback heuristic: long content is often richer than short content.
        return 1 if len(text.split()) >= 50 else 0


class HeuristicPreferenceJudge:
    """
    "Trusted judge" for demos/tests. Uses metadata quality if available.
    """

    def compare(
        self,
        prompt: str,
        left: ResponseCandidate,
        right: ResponseCandidate,
    ) -> int:
        left_score = _judge_score(left)
        right_score = _judge_score(right)
        if abs(left_score - right_score) < 1e-9:
            return 0
        return 1 if left_score > right_score else -1


class RankPreferenceJudge:
    """
    Preference judge that uses explicit metadata.rank for ground truth.
    Lower rank = better quality (rank 1 is best).
    """

    def compare(
        self,
        prompt: str,
        left: ResponseCandidate,
        right: ResponseCandidate,
    ) -> int:
        left_rank = left.metadata.get("rank")
        right_rank = right.metadata.get("rank")
        
        # If rank is missing for either candidate, fall back to heuristic
        if left_rank is None or right_rank is None:
            left_score = _judge_score(left)
            right_score = _judge_score(right)
            if abs(left_score - right_score) < 1e-9:
                return 0
            return 1 if left_score > right_score else -1
        
        # Compare ranks: lower rank is better
        if left_rank == right_rank:
            return 0
        return 1 if left_rank < right_rank else -1


class TemplateProposer:
    """
    Deterministic mutation-based proposer.
    Replace with an LLM proposer that reads (prompt, top candidates, current rubric).
    """

    def propose(
        self,
        prompt: str,
        left: ResponseCandidate,
        right: ResponseCandidate,
        rubric: Rubric,
        *,
        mode: str,
        rng: random.Random,
    ) -> Rubric:
        updated = deepcopy(rubric)
        updated.rubric_id = f"{rubric.rubric_id}_{mode}_{rng.randint(0, 9999)}"
        if mode == "raise_bar":
            self._raise_bar(updated)
        elif mode == "decompose":
            self._decompose(updated, rng)
        elif mode == "factual_focus":
            self._factual_focus(updated)
        elif mode == "anti_fluff":
            self._anti_fluff(updated)
        elif mode == "counterexample_trigger":
            self._counterexample(updated)
        else:
            self._weight_perturb(updated, rng)
        return self._renormalize(updated)

    def _raise_bar(self, rubric: Rubric) -> None:
        target = max(rubric.criteria, key=lambda c: c.weight)
        target.text = (
            target.text.rstrip(".")
            + "; for high scores, require verifiable details or concrete evidence."
        )
        target.weight *= 1.2

    def _decompose(self, rubric: Rubric, rng: random.Random) -> None:
        if not rubric.criteria:
            return
        target_idx = max(range(len(rubric.criteria)), key=lambda idx: len(rubric.criteria[idx].text))
        target = rubric.criteria[target_idx]
        if " and " in target.text:
            left_text, right_text = target.text.split(" and ", 1)
        else:
            midpoint = max(1, len(target.text) // 2)
            left_text = target.text[:midpoint].strip()
            right_text = target.text[midpoint:].strip()
        split_weight = max(0.01, target.weight / 2.0)
        rubric.criteria[target_idx] = replace(
            target,
            criterion_id=f"{target.criterion_id}_a",
            text=left_text or target.text,
            weight=split_weight,
        )
        rubric.criteria.insert(
            target_idx + 1,
            Criterion(
                criterion_id=f"{target.criterion_id}_b",
                text=right_text or target.text,
                weight=split_weight,
                criterion_type=target.criterion_type,
                check_points=target.check_points[:],
                positive_examples=target.positive_examples[:],
                negative_examples=target.negative_examples[:],
            ),
        )
        if rng.random() < 0.4:
            rubric.criteria[target_idx].check_points.append("example")

    def _factual_focus(self, rubric: Rubric) -> None:
        for criterion in rubric.criteria:
            ctype = (criterion.criterion_type or "").lower()
            if ctype in {"factual", "reasoning"}:
                criterion.weight *= 1.25
            else:
                criterion.weight *= 0.9

    def _anti_fluff(self, rubric: Rubric) -> None:
        rubric.criteria.append(
            Criterion(
                criterion_id=f"anti_fluff_{len(rubric.criteria)}",
                text="Response avoids repetitive filler and vague boilerplate.",
                weight=0.2,
                criterion_type="style",
                negative_examples=[
                    "in conclusion",
                    "overall",
                    "as an ai language model",
                ],
                check_points=["specific", "example", "step"],
            )
        )

    def _counterexample(self, rubric: Rubric) -> None:
        rubric.criteria.append(
            Criterion(
                criterion_id=f"counterexample_{len(rubric.criteria)}",
                text="If response includes claims, it should include caveats or limitations.",
                weight=0.18,
                criterion_type="reasoning",
                check_points=["however", "limit", "trade-off", "risk"],
            )
        )

    def _weight_perturb(self, rubric: Rubric, rng: random.Random) -> None:
        for criterion in rubric.criteria:
            criterion.weight = max(0.01, criterion.weight * (1 + rng.uniform(-0.15, 0.2)))

    def _renormalize(self, rubric: Rubric) -> Rubric:
        total = sum(c.weight for c in rubric.criteria)
        if total <= 0:
            return rubric
        for criterion in rubric.criteria:
            criterion.weight = criterion.weight / total
        return rubric


def _judge_score(candidate: ResponseCandidate) -> float:
    quality = candidate.metadata.get("quality")
    if quality is not None:
        return float(quality)
    source_bonus = {"strong": 0.08, "diverse": 0.05, "base": 0.0}.get(candidate.source, 0.0)
    length_bonus = min(0.08, len(candidate.text.split()) / 2000.0)
    return source_bonus + length_bonus


def _top_keywords(text: str, max_keywords: int = 4) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    keywords = [token for token, _ in ordered[:max_keywords]]
    if not keywords:
        digest = sha256(text.encode("utf-8")).hexdigest()[:8]
        return [digest]
    return keywords

