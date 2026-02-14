from __future__ import annotations

import re


def extract_tagged_segments(
    text: str,
    *,
    tag_name: str,
    case_sensitive: bool,
) -> list[str]:
    flags = re.DOTALL
    if not case_sensitive:
        flags |= re.IGNORECASE
    pattern = re.compile(
        rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>",
        flags,
    )
    matches = pattern.findall(text)
    return [match.strip() for match in matches if match.strip()]


def extract_regex_segments(
    text: str,
    *,
    compiled_pattern: re.Pattern[str],
    group: int,
) -> list[str]:
    matches = compiled_pattern.findall(text)
    if matches and isinstance(matches[0], tuple):
        return [m[group - 1].strip() for m in matches if len(m) >= group and m[group - 1].strip()]
    return [m.strip() for m in matches if isinstance(m, str) and m.strip()]
