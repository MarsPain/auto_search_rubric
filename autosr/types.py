"""Type definitions and enumerations for AutoSR.

This module provides strongly-typed enumerations to replace string-based
type identifiers, enabling better IDE support, type safety, and runtime validation.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Self


class MutationMode(Enum):
    """Mutation strategies for rubric evolution.
    
    Each mode represents a different approach to modifying a rubric
    to explore the search space effectively.
    """
    RAISE_BAR = "raise_bar"
    DECOMPOSE = "decompose"
    FACTUAL_FOCUS = "factual_focus"
    ANTI_FLUFF = "anti_fluff"
    COUNTEREXAMPLE_TRIGGER = "counterexample_trigger"
    WEIGHT_PERTURB = "weight_perturb"

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Create a MutationMode from its string representation.
        
        Args:
            value: The string value of the mutation mode.
            
        Returns:
            The corresponding MutationMode enum member.
            
        Raises:
            ValueError: If the string does not match any known mode.
        """
        try:
            return cls(value)
        except ValueError as e:
            valid_modes = [m.value for m in cls]
            raise ValueError(
                f"Unknown mutation mode: {value!r}. "
                f"Valid modes are: {valid_modes}"
            ) from e

    @classmethod
    def default_cycle(cls) -> tuple[Self, ...]:
        """Return the default cycle of mutation modes for round-robin selection.
        
        Returns:
            A tuple of all mutation modes in the standard order.
        """
        return tuple(cls)


class SearchMode(Enum):
    """Search algorithm modes for rubric optimization."""
    ITERATIVE = "iterative"
    EVOLUTIONARY = "evolutionary"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Create a SearchMode from its string representation."""
        try:
            return cls(value)
        except ValueError as e:
            valid_modes = [m.value for m in cls]
            raise ValueError(
                f"Unknown search mode: {value!r}. "
                f"Valid modes are: {valid_modes}"
            ) from e


class BackendType(Enum):
    """Backend implementation types for component creation."""
    AUTO = "auto"      # Automatically select based on configuration
    MOCK = "mock"      # Use heuristic/mock implementations
    LLM = "llm"        # Use LLM-based implementations

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Create a BackendType from its string representation."""
        try:
            return cls(value)
        except ValueError as e:
            valid_types = [t.value for t in cls]
            raise ValueError(
                f"Unknown backend type: {value!r}. "
                f"Valid types are: {valid_types}"
            ) from e


class InitializerStrategy(Enum):
    """Strategies for rubric initialization."""
    BACKEND = "backend"    # Use the configured backend (LLM or mock)
    PRESET = "preset"      # Load from preset rubrics file

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Create an InitializerStrategy from its string representation."""
        try:
            return cls(value)
        except ValueError as e:
            valid_strategies = [s.value for s in cls]
            raise ValueError(
                f"Unknown initializer strategy: {value!r}. "
                f"Valid strategies are: {valid_strategies}"
            ) from e


class ExtractionStrategy(Enum):
    """Content extraction strategies for verifier output processing."""
    IDENTITY = "identity"  # No transformation, pass through as-is
    TAG = "tag"            # Extract content from XML-style tags
    REGEX = "regex"        # Extract content using regex pattern

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Create an ExtractionStrategy from its string representation."""
        try:
            return cls(value)
        except ValueError as e:
            valid_strategies = [s.value for s in cls]
            raise ValueError(
                f"Unknown extraction strategy: {value!r}. "
                f"Valid strategies are: {valid_strategies}"
            ) from e


class LLMRole(Enum):
    """LLM component roles for model selection."""
    INITIALIZER = auto()
    PROPOSER = auto()
    VERIFIER = auto()
    JUDGE = auto()

    def __str__(self) -> str:
        return self.name.lower()
