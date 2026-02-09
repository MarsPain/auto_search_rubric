"""Prompt management package for AutoSR.

This package provides:
    - constants: Hardcoded prompt templates
    - loader: Configuration loading from YAML/JSON files
    - PromptConfig: Data class for prompt configurations
    - PromptRepository: Protocol for prompt storage backends

Usage:
    # Using constants (step 1)
    from autosr.prompts import constants as prompts
    
    system = prompts.RUBRIC_INITIALIZER_SYSTEM
    user = prompts.RUBRIC_INITIALIZER_USER_TEMPLATE.format(...)
    
    # Using external config (step 4)
    from autosr.prompts.loader import create_repository
    
    repo = create_repository("prompts/")
    config = repo.get("rubric_initializer")
    system, user = config.render(prompt_json="...", ...)
"""

from __future__ import annotations

# Re-export key classes for convenience
from .loader import (
    PromptConfig,
    PromptRepository,
    FileSystemPromptRepository,
    ConstantPromptRepository,
    create_repository,
)

__all__ = [
    # Submodules
    "constants",
    "loader",
    # Classes
    "PromptConfig",
    "PromptRepository",
    "FileSystemPromptRepository",
    "ConstantPromptRepository",
    "create_repository",
]
