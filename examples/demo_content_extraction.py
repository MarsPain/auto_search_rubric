#!/usr/bin/env python3
"""Demo script showing content extraction strategies.

This script demonstrates how to use different content extraction strategies
(TagExtractor, RegexExtractor) to extract relevant content from prompts
before passing to the underlying verifier.
"""

from __future__ import annotations

from autosr.content_extraction import (
    ContentExtractingVerifier,
    RegexExtractor,
    TagExtractor,
)
from autosr.models import Criterion, GradingProtocol, ResponseCandidate, Rubric


class DebugVerifier:
    """A verifier that prints what it receives for debugging."""

    def grade(
        self,
        prompt: str,
        candidate: ResponseCandidate,
        rubric: Rubric,
        *,
        seed: int,
    ) -> dict[str, int | None]:
        print("=" * 60)
        print("DEBUG: Prompt received by inner verifier:")
        print("=" * 60)
        print(f"Length: {len(prompt)} characters")
        print("-" * 60)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("=" * 60)
        return {"accuracy": 1, "completeness": 1}


def demo_tag_extraction() -> None:
    """Demo XML-style tag extraction."""
    full_prompt = """[|Human|]:## 角色
专业的电话纪要整理专家，擅长从通话内容中精准提炼关键信息。

## 任务
基于提供的通话记录，生成一份Markdown格式的文本总结。

## 通话记录

<通话内容>
对方：回来啦。
我：没有晚上吃什么，我还不是有饭吗？
对方：我给你留了吗？
我：哦，
</通话内容>

请直接输出文本总结：[|AI|]:"""

    candidate = ResponseCandidate(
        candidate_id="demo_candidate",
        text="晚餐食材沟通",
        source="demo",
    )

    rubric = Rubric(
        rubric_id="demo_rubric",
        criteria=[
            Criterion(criterion_id="accuracy", text="准确性", weight=0.5),
        ],
        grading_protocol=GradingProtocol(),
    )

    print("\n" + "=" * 60)
    print("STRATEGY 1: TagExtractor (XML-style tags)")
    print("=" * 60)

    # Create the extraction strategy
    extractor = TagExtractor(tag_name="通话内容")
    inner = DebugVerifier()
    verifier = ContentExtractingVerifier(inner, extractor)

    print(f"\nOriginal prompt: {len(full_prompt)} characters")
    print(f"Extracted: {len(extractor.extract(full_prompt))} characters")

    print("\nCalling verifier.grade()...")
    verifier.grade(full_prompt, candidate, rubric, seed=42)


def demo_regex_extraction() -> None:
    """Demo regex pattern extraction."""
    full_prompt = """[|System|]: You are a helpful assistant.

[|User|]: Please summarize the following conversation:
---
Alice: Hi Bob, how are you?
Bob: I'm good, thanks!
Alice: Great to hear.
---

Provide a brief summary."""

    candidate = ResponseCandidate(
        candidate_id="demo_candidate",
        text="Alice and Bob greeted each other.",
        source="demo",
    )

    rubric = Rubric(
        rubric_id="demo_rubric",
        criteria=[
            Criterion(criterion_id="accuracy", text="准确性", weight=0.5),
        ],
        grading_protocol=GradingProtocol(),
    )

    print("\n" + "=" * 60)
    print("STRATEGY 2: RegexExtractor (regex patterns)")
    print("=" * 60)

    # Create the extraction strategy - extract content between --- markers
    extractor = RegexExtractor(r"---\n(.+?)\n---", flags=re.DOTALL)
    inner = DebugVerifier()
    verifier = ContentExtractingVerifier(inner, extractor)

    print(f"\nOriginal prompt: {len(full_prompt)} characters")
    print(f"Extracted: {len(extractor.extract(full_prompt))} characters")

    print("\nCalling verifier.grade()...")
    verifier.grade(full_prompt, candidate, rubric, seed=42)


def demo_comparison() -> None:
    """Demo comparing different extraction strategies."""
    prompt_with_tags = """Instructions:
Please evaluate the response based on the conversation.

<conversation>
User: What's the weather like?
Assistant: It's sunny and 25°C today.
</conversation>

End of instructions."""

    print("\n" + "=" * 60)
    print("COMPARISON: Different extraction strategies on same prompt")
    print("=" * 60)

    # Strategy 1: No extraction
    print("\n1. IdentityExtractor (no extraction):")
    from autosr.content_extraction import IdentityExtractor
    identity = IdentityExtractor()
    result = identity.extract(prompt_with_tags)
    print(f"   Length: {len(result)} characters")
    print(f"   Contains '<conversation>': {'<conversation>' in result}")

    # Strategy 2: Tag extraction
    print("\n2. TagExtractor (tag='conversation'):")
    tag_ext = TagExtractor(tag_name="conversation")
    result = tag_ext.extract(prompt_with_tags)
    print(f"   Length: {len(result)} characters")
    print(f"   Contains '<conversation>': {'<conversation>' in result}")
    print(f"   Preview: {result[:50]}...")

    # Strategy 3: Regex extraction
    print("\n3. RegexExtractor (pattern=r'<conversation>(.+?)</conversation>'):")
    regex_ext = RegexExtractor(r"<conversation>(.+?)</conversation>")
    result = regex_ext.extract(prompt_with_tags)
    print(f"   Length: {len(result)} characters")
    print(f"   Contains '<conversation>': {'<conversation>' in result}")
    print(f"   Preview: {result[:50]}...")


import re


def main() -> None:
    """Run all demos."""
    print("\n" + "=" * 60)
    print("CONTENT EXTRACTION STRATEGIES DEMO")
    print("=" * 60)
    print("\nThis demo shows how to use different extraction strategies")
    print("to filter prompt content before passing to verifiers.")

    demo_tag_extraction()
    demo_regex_extraction()
    demo_comparison()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Available extraction strategies:

1. TagExtractor - Extract content from XML-style tags
   Example: TagExtractor("content")
   Input:  "Prefix<content>Data</content>Suffix"
   Output: "Data"

2. RegexExtractor - Extract content using regex patterns
   Example: RegexExtractor(r"Data:\\s*(.+)")
   Input:  "Header\nData: Value\nFooter"
   Output: "Value"

3. IdentityExtractor - No extraction (passthrough)
   Example: IdentityExtractor()
   Input:  "Any text"
   Output: "Any text"

Use create_content_extractor() factory or create_verifier_with_extraction()
to easily create extractors and wrapped verifiers.
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
