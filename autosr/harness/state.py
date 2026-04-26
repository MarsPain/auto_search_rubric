"""State management for harness-based search sessions.

This module provides checkpoint schema, validation, and serialization
capabilities for resumable search sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from ..data_models import Rubric


class ResumeCompatibilityError(Exception):
    """Raised when attempting to resume from an incompatible checkpoint."""
    
    def __init__(self, reason: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


class CheckpointValidationError(Exception):
    """Raised when checkpoint data fails validation."""
    
    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"Checkpoint validation failed for '{field}': {message}")
        self.field = field
        self.validation_message = message


@dataclass(slots=True)
class SearchCheckpoint:
    """Schema v1 for search session checkpoint.
    
    This dataclass represents a serializable snapshot of a search session's
    state at a specific generation. It enables recovery from interruptions
    and reproducibility of long-running searches.
    
    Attributes:
        session_id: Unique identifier for this search session
        generation: Current generation index (0-based)
        best_rubrics: Mapping from prompt_id to current best Rubric
        best_scores: Mapping from prompt_id to current best objective score
        history: Mapping from prompt_id to list of scores per generation
        scheduler_state: Serialized state from mutation scheduler (opaque dict)
        rng_state: Serialized random number generator state
        config_hash: SHA256 hash of the configuration used for this session
        dataset_hash: SHA256 hash of the dataset used for this session
        created_at_utc: ISO 8601 timestamp when checkpoint was created
        schema_version: Schema version for forward compatibility
    """
    session_id: str
    generation: int
    best_rubrics: dict[str, Rubric]
    best_scores: dict[str, float]
    history: dict[str, list[float]]
    scheduler_state: dict[str, Any]
    rng_state: dict[str, Any]
    config_hash: str
    dataset_hash: str
    algorithm_state: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "1.0"
    
    def __post_init__(self) -> None:
        # Validate required fields
        if not self.session_id.strip():
            raise CheckpointValidationError("session_id", "must not be empty")
        if self.generation < 0:
            raise CheckpointValidationError("generation", "must be >= 0")
        if not self.config_hash:
            raise CheckpointValidationError("config_hash", "must not be empty")
        if not self.dataset_hash:
            raise CheckpointValidationError("dataset_hash", "must not be empty")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "generation": self.generation,
            "best_rubrics": {
                prompt_id: rubric.to_dict()
                for prompt_id, rubric in self.best_rubrics.items()
            },
            "best_scores": self.best_scores,
            "history": self.history,
            "scheduler_state": self.scheduler_state,
            "rng_state": self.rng_state,
            "algorithm_state": self.algorithm_state,
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "created_at_utc": self.created_at_utc,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchCheckpoint":
        """Create checkpoint from dictionary representation."""
        # Handle schema version compatibility
        schema_version = data.get("schema_version", "1.0")
        if schema_version != "1.0":
            raise CheckpointValidationError(
                "schema_version",
                f"Unsupported schema version: {schema_version}. Expected: 1.0"
            )
        
        # Reconstruct Rubric objects
        best_rubrics = {
            prompt_id: Rubric.from_dict(rubric_data)
            for prompt_id, rubric_data in data.get("best_rubrics", {}).items()
        }
        
        return cls(
            session_id=data["session_id"],
            generation=data["generation"],
            best_rubrics=best_rubrics,
            best_scores=data.get("best_scores", {}),
            history=data.get("history", {}),
            scheduler_state=data.get("scheduler_state", {}),
            rng_state=data.get("rng_state", {}),
            algorithm_state=data.get("algorithm_state", {}),
            config_hash=data["config_hash"],
            dataset_hash=data["dataset_hash"],
            created_at_utc=data.get("created_at_utc", datetime.now(timezone.utc).isoformat()),
            schema_version=schema_version,
        )
    
    def to_json(self, indent: int | None = None) -> str:
        """Serialize checkpoint to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "SearchCheckpoint":
        """Deserialize checkpoint from JSON string."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise CheckpointValidationError("json", f"Invalid JSON: {e}") from e
        return cls.from_dict(data)


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute SHA256 hash of configuration dictionary.
    
    Args:
        config_dict: Configuration as serializable dictionary
        
    Returns:
        Hex digest of SHA256 hash
    """
    # Normalize by sorting keys and using consistent encoding
    normalized = json.dumps(config_dict, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_dataset_hash(dataset_path: Path) -> str:
    """Compute SHA256 hash of dataset file.
    
    Args:
        dataset_path: Path to dataset JSON file
        
    Returns:
        Hex digest of SHA256 hash
        
    Raises:
        FileNotFoundError: If dataset file does not exist
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    digest = hashlib.sha256()
    with dataset_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class ResumeValidationResult:
    """Result of resume compatibility validation."""
    compatible: bool
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_compatible(self) -> bool:
        """Check if checkpoint is compatible for resume."""
        return self.compatible


class ResumeValidator:
    """Validator for resume compatibility checks.
    
    Ensures that a checkpoint can be safely resumed by verifying:
    - Configuration matches (by hash comparison)
    - Dataset matches (by hash comparison)
    - Checkpoint data integrity
    """
    
    def __init__(
        self,
        current_config_hash: str,
        current_dataset_hash: str,
    ) -> None:
        """Initialize validator with current run parameters.
        
        Args:
            current_config_hash: Hash of the current configuration
            current_dataset_hash: Hash of the current dataset
        """
        self.current_config_hash = current_config_hash
        self.current_dataset_hash = current_dataset_hash
    
    @classmethod
    def from_config_and_dataset(
        cls,
        config_dict: dict[str, Any],
        dataset_path: Path,
    ) -> "ResumeValidator":
        """Create validator from configuration dictionary and dataset path."""
        config_hash = compute_config_hash(config_dict)
        dataset_hash = compute_dataset_hash(dataset_path)
        return cls(config_hash, dataset_hash)
    
    def validate(self, checkpoint: SearchCheckpoint) -> ResumeValidationResult:
        """Validate if a checkpoint is compatible for resume.
        
        Args:
            checkpoint: The checkpoint to validate
            
        Returns:
            Validation result with compatibility status and details
        """
        errors: dict[str, Any] = {}
        
        # Check config hash
        if checkpoint.config_hash != self.current_config_hash:
            errors["config_hash"] = {
                "checkpoint": checkpoint.config_hash,
                "current": self.current_config_hash,
            }
        
        # Check dataset hash
        if checkpoint.dataset_hash != self.current_dataset_hash:
            errors["dataset_hash"] = {
                "checkpoint": checkpoint.dataset_hash,
                "current": self.current_dataset_hash,
            }
        
        if errors:
            return ResumeValidationResult(
                compatible=False,
                reason="Configuration or dataset mismatch",
                details=errors,
            )
        
        return ResumeValidationResult(compatible=True)
    
    def validate_or_raise(self, checkpoint: SearchCheckpoint) -> None:
        """Validate checkpoint and raise exception if incompatible.
        
        Args:
            checkpoint: The checkpoint to validate
            
        Raises:
            ResumeCompatibilityError: If checkpoint is incompatible
        """
        result = self.validate(checkpoint)
        if not result.compatible:
            raise ResumeCompatibilityError(
                reason=result.reason or "Incompatible checkpoint",
                details=result.details,
            )
