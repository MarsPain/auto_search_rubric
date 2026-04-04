#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"

REQUIRED_FILES = (
    "AGENTS.md",
    "docs/DESIGN.md",
    "docs/FRONTEND.md",
    "docs/PLANS.md",
    "docs/PRODUCT_SENSE.md",
    "docs/ROADMAP.md",
    "docs/design-docs/01-architecture.md",
    "docs/generated/README.md",
    "docs/product-specs/README.md",
    "docs/references/README.md",
)

REQUIRED_DIRS = (
    "docs/design-docs",
    "docs/exec-plans/active",
    "docs/exec-plans/completed",
    "docs/exec-plans/tech-debt",
    "docs/generated",
    "docs/product-specs",
    "docs/references",
)

REDIRECT_FILES = (
    "ROADMAP.md",
    "ROADMAP_ARCHITECTURE.md",
    "STAGE0_IMPLEMENTATION.md",
    "STAGE1_IMPLEMENTATION.md",
)

CORE_PLAN_FOLDERS = ("active", "completed", "tech-debt")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _to_abs(rel_path: str) -> Path:
    return REPO_ROOT / rel_path


def _collect_markdown_files() -> list[Path]:
    top_level = [
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / "ROADMAP.md",
        REPO_ROOT / "ROADMAP_ARCHITECTURE.md",
        REPO_ROOT / "STAGE0_IMPLEMENTATION.md",
        REPO_ROOT / "STAGE1_IMPLEMENTATION.md",
    ]
    return sorted(top_level + list(DOCS_ROOT.rglob("*.md")))


def _is_external_link(target: str) -> bool:
    lowered = target.lower()
    return (
        lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("mailto:")
        or lowered.startswith("tel:")
    )


def _validate_paths(errors: list[str]) -> None:
    for rel in REQUIRED_FILES:
        if not _to_abs(rel).exists():
            errors.append(f"missing required file: {rel}")

    for rel in REQUIRED_DIRS:
        if not _to_abs(rel).is_dir():
            errors.append(f"missing required directory: {rel}")

    plans_root = DOCS_ROOT / "exec-plans"
    for folder in CORE_PLAN_FOLDERS:
        if not (plans_root / folder).exists():
            errors.append(f"missing plan bucket: docs/exec-plans/{folder}")

    completed_plans = list((plans_root / "completed").glob("*.md"))
    if not completed_plans:
        errors.append("docs/exec-plans/completed must contain at least one .md plan")


def _validate_redirect_docs(errors: list[str]) -> None:
    for rel in REDIRECT_FILES:
        path = _to_abs(rel)
        if not path.exists():
            errors.append(f"missing redirect file: {rel}")
            continue
        content = path.read_text(encoding="utf-8")
        if "已迁移" not in content:
            errors.append(f"redirect file missing migration marker '已迁移': {rel}")
        if "(docs/" not in content:
            errors.append(f"redirect file missing docs link target: {rel}")


def _validate_agents_constraints(errors: list[str]) -> None:
    agents = REPO_ROOT / "AGENTS.md"
    if not agents.exists():
        errors.append("AGENTS.md not found")
        return
    line_count = len(agents.read_text(encoding="utf-8").splitlines())
    if line_count > 140:
        errors.append(f"AGENTS.md should stay concise (<=140 lines), got {line_count}")


def _validate_markdown_links(errors: list[str]) -> None:
    for md_file in _collect_markdown_files():
        lines = md_file.read_text(encoding="utf-8").splitlines()
        in_fence = False
        searchable_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if not in_fence:
                searchable_lines.append(line)

        for raw_target in MARKDOWN_LINK_RE.findall("\n".join(searchable_lines)):
            target = raw_target.strip()
            if not target or target.startswith("#") or _is_external_link(target):
                continue

            target = target.split("#", 1)[0].strip()
            if not target:
                continue

            candidate = (md_file.parent / target).resolve()
            try:
                candidate.relative_to(REPO_ROOT)
            except ValueError:
                errors.append(
                    f"link escapes repository root: {md_file.relative_to(REPO_ROOT)} -> {raw_target}"
                )
                continue

            if not candidate.exists():
                errors.append(
                    f"broken link: {md_file.relative_to(REPO_ROOT)} -> {raw_target}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate docs context structure and links.")
    parser.add_argument("--verbose", action="store_true", help="Print additional details.")
    args = parser.parse_args()

    errors: list[str] = []
    _validate_paths(errors)
    _validate_redirect_docs(errors)
    _validate_agents_constraints(errors)
    _validate_markdown_links(errors)

    if errors:
        print("Docs validation failed:")
        for item in errors:
            print(f"- {item}")
        return 1

    if args.verbose:
        print(f"Validated {len(_collect_markdown_files())} markdown files.")
    print("Docs validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
