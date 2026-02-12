"""Tests for content extraction strategies and verifier wrapper."""

from __future__ import annotations

import re
import unittest

from autosr.content_extraction import (
    ContentExtractingVerifier,
    IdentityExtractor,
    RegexExtractor,
    TagExtractor,
    create_content_extractor,
    create_verifier_with_extraction,
)
from autosr.models import Criterion, GradingProtocol, ResponseCandidate, Rubric


class MockVerifier:
    """Mock verifier for testing that records the prompt it receives."""

    def __init__(self, fixed_grades: dict[str, int | None] | None = None) -> None:
        self.last_prompt: str | None = None
        self.last_candidate: ResponseCandidate | None = None
        self.fixed_grades = fixed_grades or {"c1": 1}

    def grade(
        self,
        prompt: str,
        candidate: ResponseCandidate,
        rubric: Rubric,
        *,
        seed: int,
    ) -> dict[str, int | None]:
        self.last_prompt = prompt
        self.last_candidate = candidate
        return dict(self.fixed_grades)


class TestTagExtractor(unittest.TestCase):
    """Test cases for TagExtractor strategy."""

    def test_extracts_single_tag(self) -> None:
        """Test extraction from a single tag pair."""
        extractor = TagExtractor("content")
        text = "Prefix<content>Inner content</content>Suffix"
        result = extractor.extract(text)
        self.assertEqual(result, "Inner content")

    def test_extracts_multiple_tags(self) -> None:
        """Test joining multiple tag matches."""
        extractor = TagExtractor("content")
        text = "<content>First</content><content>Second</content>"
        result = extractor.extract(text)
        self.assertEqual(result, "First\n\nSecond")

    def test_returns_original_when_no_tags(self) -> None:
        """Test original text returned when no matching tags."""
        extractor = TagExtractor("content")
        text = "Plain text without tags"
        result = extractor.extract(text)
        self.assertEqual(result, text)

    def test_case_insensitive_matching(self) -> None:
        """Test tag matching is case-insensitive by default."""
        extractor = TagExtractor("content")
        text = "<CONTENT>Upper</CONTENT><content>Lower</content>"
        result = extractor.extract(text)
        self.assertIn("Upper", result)
        self.assertIn("Lower", result)

    def test_case_sensitive_matching(self) -> None:
        """Test case-sensitive matching when enabled."""
        extractor = TagExtractor("CONTENT", case_sensitive=True)
        text = "<CONTENT>Upper</CONTENT><content>Lower</content>"
        result = extractor.extract(text)
        self.assertIn("Upper", result)
        self.assertNotIn("Lower", result)

    def test_strips_whitespace(self) -> None:
        """Test whitespace is stripped from extracted content."""
        extractor = TagExtractor("content")
        text = "<content>\n  Inner  \n</content>"
        result = extractor.extract(text)
        self.assertEqual(result, "Inner")

    def test_custom_join_separator(self) -> None:
        """Test custom separator for multiple matches."""
        extractor = TagExtractor("content", join_multiple=" | ")
        text = "<content>A</content><content>B</content>"
        result = extractor.extract(text)
        self.assertEqual(result, "A | B")

    def test_empty_tags_returns_original(self) -> None:
        """Test that original text is returned when tags are empty."""
        extractor = TagExtractor("content")
        text = "Prefix<content>   </content>Suffix"
        result = extractor.extract(text)
        self.assertEqual(result, text)

    def test_tag_name_property(self) -> None:
        """Test that tag_name property returns the configured tag."""
        extractor = TagExtractor("custom_tag")
        self.assertEqual(extractor.tag_name, "custom_tag")


class TestRegexExtractor(unittest.TestCase):
    """Test cases for RegexExtractor strategy."""

    def test_extracts_single_match(self) -> None:
        """Test extraction using regex pattern."""
        extractor = RegexExtractor(r"Data:\s*(.+?)(?:\n|$)")
        text = "Title\nData: Hello World\nFooter"
        result = extractor.extract(text)
        self.assertEqual(result, "Hello World")

    def test_extracts_multiple_matches(self) -> None:
        """Test joining multiple regex matches."""
        extractor = RegexExtractor(r"Item:\s*(.+?)(?:\n|$)")
        text = "Item: First\nItem: Second\n"
        result = extractor.extract(text)
        self.assertEqual(result, "First\n\nSecond")

    def test_returns_original_when_no_match(self) -> None:
        """Test original text returned when pattern doesn't match."""
        extractor = RegexExtractor(r"Missing:\s*(.+)")
        text = "No matching pattern here"
        result = extractor.extract(text)
        self.assertEqual(result, text)

    def test_custom_group_extraction(self) -> None:
        """Test extracting from a specific capture group."""
        extractor = RegexExtractor(r"(\w+):\s*(.+)", group=2)
        text = "Key: Value"
        result = extractor.extract(text)
        self.assertEqual(result, "Value")

    def test_multiline_matching(self) -> None:
        """Test multiline content extraction with DOTALL."""
        extractor = RegexExtractor(r"<block>(.+?)</block>")
        text = "<block>Line 1\nLine 2\nLine 3</block>"
        result = extractor.extract(text)
        self.assertEqual(result, "Line 1\nLine 2\nLine 3")

    def test_custom_separator(self) -> None:
        """Test custom separator for multiple matches."""
        extractor = RegexExtractor(r"\d+:\s*(\w+)", join_multiple=", ")
        text = "1: First\n2: Second\n3: Third"
        result = extractor.extract(text)
        self.assertEqual(result, "First, Second, Third")


class TestIdentityExtractor(unittest.TestCase):
    """Test cases for IdentityExtractor (no-op strategy)."""

    def test_returns_text_unchanged(self) -> None:
        """Test that text is returned unchanged."""
        extractor = IdentityExtractor()
        text = "Any text content"
        result = extractor.extract(text)
        self.assertEqual(result, text)

    def test_handles_empty_string(self) -> None:
        """Test handling of empty string."""
        extractor = IdentityExtractor()
        result = extractor.extract("")
        self.assertEqual(result, "")


class TestContentExtractingVerifier(unittest.TestCase):
    """Test cases for ContentExtractingVerifier."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.mock_verifier = MockVerifier()
        self.candidate = ResponseCandidate(
            candidate_id="test_1",
            text="Test response",
            source="test",
        )
        self.rubric = Rubric(
            rubric_id="test_rubric",
            criteria=[
                Criterion(
                    criterion_id="c1",
                    text="Test criterion",
                    weight=1.0,
                ),
            ],
            grading_protocol=GradingProtocol(),
        )

    def test_extracts_content_before_grading(self) -> None:
        """Test that content is extracted before passing to inner verifier."""
        extractor = TagExtractor("content")
        verifier = ContentExtractingVerifier(self.mock_verifier, extractor)

        prompt = "Prefix<content>Extracted</content>Suffix"
        result = verifier.grade(prompt, self.candidate, self.rubric, seed=42)

        self.assertEqual(result, {"c1": 1})
        self.assertEqual(self.mock_verifier.last_prompt, "Extracted")

    def test_no_extraction_when_no_tags(self) -> None:
        """Test original prompt passed when extraction finds nothing."""
        extractor = TagExtractor("nonexistent")
        verifier = ContentExtractingVerifier(self.mock_verifier, extractor)

        prompt = "Plain prompt without tags"
        verifier.grade(prompt, self.candidate, self.rubric, seed=42)

        self.assertEqual(self.mock_verifier.last_prompt, prompt)

    def test_preserves_candidate_and_rubric(self) -> None:
        """Test that candidate and rubric are passed unchanged."""
        extractor = TagExtractor("content")
        verifier = ContentExtractingVerifier(self.mock_verifier, extractor)

        prompt = "<content>Test</content>"
        verifier.grade(prompt, self.candidate, self.rubric, seed=42)

        self.assertIs(self.mock_verifier.last_candidate, self.candidate)

    def test_inner_verifier_property(self) -> None:
        """Test that inner_verifier property returns the wrapped verifier."""
        extractor = TagExtractor("content")
        verifier = ContentExtractingVerifier(self.mock_verifier, extractor)
        self.assertIs(verifier.inner_verifier, self.mock_verifier)

    def test_extractor_property(self) -> None:
        """Test that extractor property returns the configured extractor."""
        extractor = TagExtractor("custom")
        verifier = ContentExtractingVerifier(self.mock_verifier, extractor)
        self.assertIs(verifier.extractor, extractor)

    def test_with_regex_extractor(self) -> None:
        """Test using RegexExtractor strategy."""
        extractor = RegexExtractor(r"Content:\s*(.+?)(?:\n|$)")
        verifier = ContentExtractingVerifier(self.mock_verifier, extractor)

        prompt = "Header\nContent: Important data\nFooter"
        verifier.grade(prompt, self.candidate, self.rubric, seed=42)

        self.assertEqual(self.mock_verifier.last_prompt, "Important data")


class TestCreateContentExtractor(unittest.TestCase):
    """Test cases for create_content_extractor factory function."""

    def test_creates_identity_extractor(self) -> None:
        """Test creating identity extractor."""
        result = create_content_extractor("identity")
        self.assertIsInstance(result, IdentityExtractor)

    def test_none_creates_identity_extractor(self) -> None:
        """Test that None strategy creates identity extractor."""
        result = create_content_extractor(None)
        self.assertIsInstance(result, IdentityExtractor)

    def test_creates_tag_extractor(self) -> None:
        """Test creating tag extractor with required args."""
        result = create_content_extractor("tag", tag_name="content")
        self.assertIsInstance(result, TagExtractor)
        self.assertEqual(result.tag_name, "content")

    def test_creates_xml_extractor(self) -> None:
        """Test that 'xml' is alias for 'tag'."""
        result = create_content_extractor("xml", tag_name="data")
        self.assertIsInstance(result, TagExtractor)
        self.assertEqual(result.tag_name, "data")

    def test_tag_extractor_requires_tag_name(self) -> None:
        """Test that tag strategy requires tag_name argument."""
        with self.assertRaises(ValueError) as ctx:
            create_content_extractor("tag")
        self.assertIn("tag_name", str(ctx.exception))

    def test_creates_regex_extractor(self) -> None:
        """Test creating regex extractor with required args."""
        result = create_content_extractor("regex", pattern=r"test:\s*(.+)")
        self.assertIsInstance(result, RegexExtractor)

    def test_regex_extractor_requires_pattern(self) -> None:
        """Test that regex strategy requires pattern argument."""
        with self.assertRaises(ValueError) as ctx:
            create_content_extractor("regex")
        self.assertIn("pattern", str(ctx.exception))

    def test_unknown_strategy_raises_error(self) -> None:
        """Test that unknown strategy raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_content_extractor("unknown")
        self.assertIn("Unknown extraction strategy", str(ctx.exception))


class TestCreateVerifierWithExtraction(unittest.TestCase):
    """Test cases for create_verifier_with_extraction convenience function."""

    def test_returns_original_verifier_for_identity(self) -> None:
        """Test that original verifier is returned for identity strategy."""
        mock = MockVerifier()
        result = create_verifier_with_extraction(mock, "identity")
        self.assertIs(result, mock)

    def test_returns_original_verifier_for_none(self) -> None:
        """Test that original verifier is returned for None strategy."""
        mock = MockVerifier()
        result = create_verifier_with_extraction(mock, None)
        self.assertIs(result, mock)

    def test_returns_wrapped_verifier_for_tag(self) -> None:
        """Test that wrapped verifier is returned for tag strategy."""
        mock = MockVerifier()
        result = create_verifier_with_extraction(mock, "tag", tag_name="content")
        self.assertIsInstance(result, ContentExtractingVerifier)
        self.assertIs(result.inner_verifier, mock)

    def test_returns_wrapped_verifier_for_regex(self) -> None:
        """Test that wrapped verifier is returned for regex strategy."""
        mock = MockVerifier()
        result = create_verifier_with_extraction(mock, "regex", pattern=r"test")
        self.assertIsInstance(result, ContentExtractingVerifier)


class TestRealWorldExamples(unittest.TestCase):
    """Test cases using realistic dataset formats."""

    def test_call_summary_extraction(self) -> None:
        """Test extraction from call summary dataset format."""
        extractor = TagExtractor("通话内容")

        prompt = """[|Human|]:## 角色
专业的电话纪要整理专家

## 任务
基于提供的通话记录，生成一份Markdown格式的文本总结。

## 通话记录

<通话内容>
对方：回来啦。
我：没有晚上吃什么，我还不是有饭吗？
对方：我给你留了吗？
</通话内容>

请直接输出文本总结：[|AI|]:"""

        result = extractor.extract(prompt)

        # Verify only content within tags is extracted
        self.assertIn("对方：回来啦。", result)
        self.assertIn("我：没有晚上吃什么", result)
        # Verify instructions are NOT in extracted content
        self.assertNotIn("## 角色", result)
        self.assertNotIn("## 任务", result)

    def test_multiple_content_sections(self) -> None:
        """Test extracting multiple content sections."""
        extractor = TagExtractor("section", join_multiple="\n---\n")

        prompt = """<section>
第一部分内容
</section>

中间的其他文本

<section>
第二部分内容
</section>"""

        result = extractor.extract(prompt)
        self.assertIn("第一部分内容", result)
        self.assertIn("第二部分内容", result)
        self.assertIn("\n---\n", result)


if __name__ == "__main__":
    unittest.main()
