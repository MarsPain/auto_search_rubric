"""Prompt constants for LLM components.

This module contains hardcoded prompt templates as constants.
These can be overridden by external YAML/JSON configurations using PromptLoader.
"""

from __future__ import annotations

# Required criterion fields for validation
_REQUIRED_CRITERION_FIELDS = ["criterion_id", "text", "weight"]

# =============================================================================
# Rubric Initializer Prompts
# =============================================================================

RUBRIC_INITIALIZER_SYSTEM = (
    "You are a rubric designer. Return ONLY one JSON object. "
    "Required top-level keys: rubric_id, criteria, grading_protocol. "
    "Each criterion object MUST include non-empty criterion_id and text, plus weight. "
    "Do not omit required fields. Do not return markdown, comments, or partial patches."
)

RUBRIC_INITIALIZER_USER_TEMPLATE = """{{
    "task": "Create an evaluation rubric for the prompt.",
    "prompt": {prompt_json},
    "candidate_samples": {candidate_samples_json},
    "constraints": {{
        "criteria_count_range": [3, 6],
        "weights_sum_hint": 1.0,
        "criterion_required_fields": {required_fields_json},
        "return_full_rubric": true,
        "grading_protocol_default": {{
            "num_votes": 1,
            "allow_na": true,
            "vote_method": "majority"
        }}
    }}
}}"""

# =============================================================================
# Rubric Proposer (Mutator) Prompts
# =============================================================================

RUBRIC_PROPOSER_SYSTEM = (
    "You mutate rubrics for ranking response quality. Return ONLY one JSON rubric object. "
    "Required top-level keys: rubric_id, criteria; grading_protocol is optional. "
    "Each criterion object MUST include non-empty criterion_id and text, plus weight. "
    "Return a full rubric, not a delta/patch. Keep criteria precise and scoreable. "
    "You MUST strictly follow output_requirements.mode_requirements for the selected mode."
)

RUBRIC_PROPOSER_USER_TEMPLATE = """{{
    "task": "Mutate the rubric according to mode while preserving intent.",
    "mode": "{mode}",
    "prompt": {prompt_json},
    "current_rubric": {current_rubric_json},
    "top_candidate": {top_candidate_json},
    "runner_up_candidate": {runner_up_candidate_json},
    "output_requirements": {{
        "must_return_full_rubric": true,
        "criterion_required_fields": {required_fields_json},
        "preserve_criterion_text_even_if_unchanged": true,
        "weights_must_sum_to_one": true,
        "mode_requirements": {{
            "raise_bar": {{
                "must_tighten_existing_criterion": true,
                "must_raise_target_weight_min_ratio": 1.1,
                "should_keep_criteria_count_delta_within": 1
            }},
            "decompose": {{
                "must_split_one_criterion": true,
                "split_min_children": 2,
                "must_keep_semantic_coverage": true,
                "split_weight_sum_tolerance": 0.05
            }},
            "factual_focus": {{
                "must_increase_factual_or_reasoning_weight": true,
                "must_decrease_or_keep_non_factual_weight": true,
                "must_keep_criteria_count_delta_within": 1
            }},
            "anti_fluff": {{
                "must_add_or_strengthen_anti_fluff_criterion": true,
                "must_include_negative_examples": true
            }},
            "counterexample_trigger": {{
                "must_add_or_strengthen_counterexample_criterion": true,
                "must_require_limitations_or_tradeoffs": true
            }},
            "weight_perturb": {{
                "must_preserve_criterion_text": true,
                "must_keep_criterion_set_stable": true,
                "weight_change_ratio_range": [0.8, 1.2]
            }}
        }}
    }}
}}"""

# =============================================================================
# Verifier Prompts
# =============================================================================

VERIFIER_SYSTEM = (
    "You are a strict evaluator. Return ONLY JSON: "
    '{{"grades": {{"criterion_id": score|null, ...}}}}. '
    "Use continuous scores in [0, 5] (or [0,1] if needed)."
)

VERIFIER_USER_TEMPLATE = """{{
    "seed": {seed},
    "prompt": {prompt_json},
    "candidate": {candidate_json},
    "criteria": {criteria_json},
    "allow_na": {allow_na},
    "grade_scale_max": {grade_scale_max}
}}"""

# =============================================================================
# Preference Judge Prompts
# =============================================================================

JUDGE_SYSTEM = (
    "Compare two responses and return ONLY JSON: "
    '{{"preference": 1|-1|0}} where 1 means left is preferred.'
)

JUDGE_USER_TEMPLATE = """{{
    "prompt": {prompt_json},
    "left": {left_json},
    "right": {right_json}
}}"""


# =============================================================================
# Template Metadata
# =============================================================================

TEMPLATE_REGISTRY: dict[str, dict[str, list[str]]] = {
    "rubric_initializer": {
        "variables": ["prompt_json", "candidate_samples_json", "required_fields_json"],
        "system": RUBRIC_INITIALIZER_SYSTEM,
        "user_template": RUBRIC_INITIALIZER_USER_TEMPLATE,
    },
    "rubric_proposer": {
        "variables": [
            "mode",
            "prompt_json",
            "current_rubric_json",
            "top_candidate_json",
            "runner_up_candidate_json",
            "required_fields_json",
        ],
        "system": RUBRIC_PROPOSER_SYSTEM,
        "user_template": RUBRIC_PROPOSER_USER_TEMPLATE,
    },
    "verifier": {
        "variables": [
            "seed",
            "prompt_json",
            "candidate_json",
            "criteria_json",
            "allow_na",
            "grade_scale_max",
        ],
        "system": VERIFIER_SYSTEM,
        "user_template": VERIFIER_USER_TEMPLATE,
    },
    "judge": {
        "variables": ["prompt_json", "left_json", "right_json"],
        "system": JUDGE_SYSTEM,
        "user_template": JUDGE_USER_TEMPLATE,
    },
}


def get_required_fields_json() -> str:
    """Return JSON-encoded required criterion fields."""
    import json

    return json.dumps(_REQUIRED_CRITERION_FIELDS)
