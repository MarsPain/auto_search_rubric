"""Harness module for resumable, observable search sessions.

This module provides the harness layer for AutoSR search operations,
enabling checkpoint/resume capabilities without modifying core algorithm logic.

Stage 0 (Completed): API surface and schema definition
- SearchSession: Session lifecycle wrapper
- SearchCheckpoint: Serializable checkpoint schema v1
- ResumeValidator: Configuration/dataset compatibility validation

Stage 1 (Current): Full checkpoint/resume implementation
- StateManager: Persistent storage for checkpoints with atomic writes
- SearchSession.resume(): Resume from checkpoint
- SearchSession.run_step(): Step-wise execution with per-generation checkpointing
- CLI support: --resume-from, --checkpoint-every-generation, --checkpoint-interval-seconds
"""

from __future__ import annotations

# Session management
from .session import SearchSession, SessionStateError, StepResult

# State management and validation
from .state import (
    CheckpointValidationError,
    ResumeCompatibilityError,
    ResumeValidationResult,
    ResumeValidator,
    SearchCheckpoint,
    compute_config_hash,
    compute_dataset_hash,
)

# Storage
from .storage import (
    CheckpointCorruptedError,
    CheckpointMetadata,
    CheckpointNotFoundError,
    StateManager,
)

__all__ = [
    # Session
    "SearchSession",
    "SessionStateError",
    "StepResult",
    # State/Checkpoint
    "SearchCheckpoint",
    "CheckpointValidationError",
    "ResumeCompatibilityError",
    "ResumeValidationResult",
    "ResumeValidator",
    "compute_config_hash",
    "compute_dataset_hash",
    # Storage
    "StateManager",
    "CheckpointMetadata",
    "CheckpointNotFoundError",
    "CheckpointCorruptedError",
]
