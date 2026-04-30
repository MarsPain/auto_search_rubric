"""Search session lifecycle management for harness-based execution.

This module provides the SearchSession wrapper that adds session management
capabilities to existing search algorithms without modifying their core logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..data_models import PromptExample
from ..interfaces import Searcher, SteppableSearcher
from ..search.config import SearchResult
from .state import (
    ResumeCompatibilityError,
    SearchCheckpoint,
    compute_config_hash,
    compute_dataset_hash,
)
from .storage import CheckpointSaveError

if TYPE_CHECKING:
    from ..config import RuntimeConfig
    from ..factory import ComponentFactory
    from .storage import StateManager

logger = logging.getLogger("reward_harness.harness")


class SessionStateError(Exception):
    """Raised when session is in an invalid state for an operation."""

    pass


@dataclass
class StepResult:
    """Result from a single search step.

    Attributes:
        generation: Current generation index
        is_finished: Whether search has completed
        best_scores: Current best scores per prompt
        checkpoint_path: Path to checkpoint if saved, None otherwise
    """

    generation: int
    is_finished: bool
    best_scores: dict[str, float]
    checkpoint_path: Path | None = None


class SearchSession:
    """Harness session wrapper for search algorithms.

    SearchSession provides lifecycle management for search operations while
    delegating actual search logic to existing searcher implementations.
    This is a thin wrapper that does not modify core algorithm behavior.

    Usage (Stage 0/1 - basic wrapping and checkpointing):
        session = SearchSession.create(
            prompts=prompts,
            config=config,
            factory=factory,
            state_manager=state_manager,
            checkpoint_every_generation=True,
        )
        result = session.run_to_completion()

    Usage (Stage 1 - step-wise execution with checkpointing):
        session = SearchSession.create(
            prompts=prompts,
            config=config,
            factory=factory,
            state_manager=state_manager,
            checkpoint_every_generation=True,
        )
        while not session.is_finished():
            step_result = session.run_step()
            print(f"Generation {step_result.generation} complete")

    Usage (Stage 1 - resume from checkpoint):
        session = SearchSession.resume(
            resume_from="session_20260402_001",  # or checkpoint path
            config=config,
            factory=factory,
            state_manager=state_manager,
            prompts=prompts,
            dataset_path=dataset_path,
        )
        result = session.run_to_completion()
    """

    def __init__(
        self,
        session_id: str,
        prompts: list[PromptExample],
        searcher: Searcher,
        config: RuntimeConfig,
        factory: ComponentFactory,
        state_manager: StateManager | None = None,
        checkpoint_every_generation: bool = False,
        checkpoint_interval_seconds: float | None = None,
        dataset_path: Path | None = None,
    ) -> None:
        """Initialize search session.

        Args:
            session_id: Unique identifier for this session
            prompts: List of prompt examples to search over
            searcher: The underlying search algorithm instance
            config: Runtime configuration
            factory: Component factory for creating search components
            state_manager: Optional state manager for checkpoint persistence
            checkpoint_every_generation: Whether to checkpoint after each generation
            checkpoint_interval_seconds: Optional time-based checkpoint interval
        """
        self._session_id = session_id
        self._prompts = prompts
        self._searcher = searcher
        self._config = config
        self._factory = factory
        self._state_manager = state_manager
        self._checkpoint_every_generation = checkpoint_every_generation
        self._checkpoint_interval_seconds = checkpoint_interval_seconds
        self._dataset_path = dataset_path

        # Session state tracking
        self._current_generation = 0
        self._finished = False
        self._result: SearchResult | None = None
        self._is_resumed = False
        self._resume_source: str | None = None
        self._resume_semantics: str | None = None
        self._last_checkpoint_time: datetime | None = None

        # For step-wise execution (Stage 1)
        self._step_state: dict[str, Any] | None = None
        self._initialized = False

        logger.info(
            "SearchSession initialized session_id=%s mode=%s checkpoint_enabled=%s",
            session_id,
            config.search.mode.value
            if hasattr(config.search.mode, "value")
            else str(config.search.mode),
            state_manager is not None,
        )

    @property
    def session_id(self) -> str:
        """Get session identifier."""
        return self._session_id

    @property
    def current_generation(self) -> int:
        """Get current generation index."""
        return self._current_generation

    @property
    def is_resumed(self) -> bool:
        """Check if this session was resumed from a checkpoint."""
        return self._is_resumed

    @property
    def resume_source(self) -> str | None:
        """Get the source checkpoint identifier if resumed."""
        return self._resume_source

    @classmethod
    def create(
        cls,
        prompts: list[PromptExample],
        config: RuntimeConfig,
        factory: ComponentFactory,
        session_id: str | None = None,
        state_manager: StateManager | None = None,
        checkpoint_every_generation: bool = False,
        checkpoint_interval_seconds: float | None = None,
        dataset_path: Path | None = None,
    ) -> "SearchSession":
        """Create a new search session.

        This is the factory method for creating fresh search sessions.
        The session_id is generated if not provided.

        Args:
            prompts: List of prompt examples to search over
            config: Runtime configuration
            factory: Component factory for creating search components
            session_id: Optional session identifier (auto-generated if None)
            state_manager: Optional state manager for checkpoint persistence
            checkpoint_every_generation: Whether to checkpoint after each generation
            checkpoint_interval_seconds: Optional time-based checkpoint interval
            dataset_path: Optional path to dataset for checkpoint hash

        Returns:
            A new SearchSession instance
        """
        from datetime import datetime, timezone

        # Generate session_id if not provided
        if session_id is None:
            session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")

        # Create the searcher through factory
        searcher = factory.create_searcher(prompts)

        return cls(
            session_id=session_id,
            prompts=prompts,
            searcher=searcher,
            config=config,
            factory=factory,
            state_manager=state_manager,
            checkpoint_every_generation=checkpoint_every_generation,
            checkpoint_interval_seconds=checkpoint_interval_seconds,
            dataset_path=dataset_path,
        )

    @classmethod
    def resume(
        cls,
        resume_from: str | Path,
        config: RuntimeConfig,
        factory: ComponentFactory,
        state_manager: StateManager,
        prompts: list[PromptExample] | None = None,
        dataset_path: Path | None = None,
        checkpoint_every_generation: bool = False,
        checkpoint_interval_seconds: float | None = None,
    ) -> "SearchSession":
        """Resume a search session from a checkpoint.

        Args:
            resume_from: Session ID or path to checkpoint file
            config: Runtime configuration (must match checkpoint)
            factory: Component factory for creating search components
            state_manager: State manager for loading checkpoint (required)
            prompts: Optional prompts list (must match checkpoint dataset)
            dataset_path: Path to dataset for hash validation
            checkpoint_every_generation: Whether to checkpoint after each generation
            checkpoint_interval_seconds: Optional time-based checkpoint interval

        Returns:
            A resumed SearchSession instance

        Raises:
            ResumeCompatibilityError: If checkpoint is incompatible with config/dataset
            CheckpointNotFoundError: If checkpoint not found
        """
        from .storage import CheckpointNotFoundError

        # Validate prompts early
        if prompts is None:
            raise ValueError("prompts required when resuming")

        # Load checkpoint
        try:
            if Path(resume_from).exists():
                checkpoint = state_manager.load_checkpoint(path=Path(resume_from))
            else:
                checkpoint = state_manager.load_checkpoint(session_id=str(resume_from))
        except CheckpointNotFoundError:
            raise

        # Validate compatibility
        # Dataset hash must always match
        if dataset_path is not None:
            current_dataset_hash = compute_dataset_hash(dataset_path)
            if (
                checkpoint.dataset_hash != "unknown"
                and checkpoint.dataset_hash != current_dataset_hash
            ):
                raise ResumeCompatibilityError(
                    reason="Dataset mismatch",
                    details={
                        "checkpoint_dataset_hash": checkpoint.dataset_hash,
                        "current_dataset_hash": current_dataset_hash,
                    },
                )

        # Config hash mismatch is a warning (allows changing parameters like generations)
        current_config_hash = compute_config_hash(_config_to_dict(config))
        if checkpoint.config_hash != current_config_hash:
            logger.warning(
                "Config hash mismatch when resuming session_id=%s. "
                "This is expected if you changed search parameters (e.g., generations). "
                "checkpoint_hash=%s current_hash=%s",
                checkpoint.session_id,
                checkpoint.config_hash[:8] + "...",
                current_config_hash[:8] + "...",
            )

        # Create searcher
        searcher = factory.create_searcher(prompts)

        # Restore checkpoint-dependent runtime state as best effort.
        rng_restored = _restore_rng_state(searcher, checkpoint.rng_state)
        scheduler_restored = _restore_scheduler_state(
            searcher, checkpoint.scheduler_state
        )

        # Create session
        session = cls(
            session_id=checkpoint.session_id,
            prompts=prompts,
            searcher=searcher,
            config=config,
            factory=factory,
            state_manager=state_manager,
            checkpoint_every_generation=checkpoint_every_generation,
            checkpoint_interval_seconds=checkpoint_interval_seconds,
            dataset_path=dataset_path,
        )

        # Restore session state
        session._current_generation = checkpoint.generation
        session._is_resumed = True
        session._resume_source = str(resume_from)
        session._last_checkpoint_time = _parse_checkpoint_timestamp(
            checkpoint.created_at_utc
        )

        resume_semantics = "reseed_from_checkpoint"

        # Initialize step state for continued execution.
        if isinstance(searcher, SteppableSearcher) and searcher.supports_step_execution:
            restored_step_state = searcher.restore_algorithm_state(
                checkpoint.algorithm_state,
                prompt_ids=[item.prompt_id for item in prompts],
                best_rubrics=checkpoint.best_rubrics,
                best_scores=checkpoint.best_scores,
                history=checkpoint.history,
            )
            if restored_step_state is not None:
                session._step_state = restored_step_state
                session._initialized = True
                resume_semantics = "continue_from_checkpoint"
            else:
                # Legacy checkpoint fallback (no algorithm_state): re-seed state from scratch.
                step_state = searcher.initialize_step_state(prompts)
                step_state["best_rubrics"].update(checkpoint.best_rubrics)
                step_state["best_scores"].update(checkpoint.best_scores)
                for pid, scores in checkpoint.history.items():
                    if pid in step_state["history"]:
                        step_state["history"][pid] = list(scores)
                session._step_state = step_state
                session._initialized = True
        else:
            session._step_state = {
                "best_rubrics": checkpoint.best_rubrics,
                "best_scores": checkpoint.best_scores,
                "history": checkpoint.history,
            }
            session._initialized = False

        if not rng_restored:
            logger.warning(
                "RNG state was not fully restored for session_id=%s; resumed path may drift",
                checkpoint.session_id,
            )
        if not scheduler_restored:
            logger.warning(
                "Scheduler state was not fully restored for session_id=%s; resumed path may drift",
                checkpoint.session_id,
            )

        session._resume_semantics = resume_semantics

        logger.info(
            "SearchSession resumed session_id=%s generation=%d source=%s semantics=%s",
            checkpoint.session_id,
            checkpoint.generation,
            resume_from,
            resume_semantics,
        )

        return session

    def run_to_completion(self) -> SearchResult:
        """Run search to completion.

        For iterative mode, delegates directly to searcher's search().
        For evolutionary mode with checkpointing, uses step-wise execution.

        Returns:
            SearchResult containing best rubrics, scores, and history

        Raises:
            SessionStateError: If session has already finished
        """
        if self._finished:
            raise SessionStateError(
                "Session has already finished. Use get_result() to retrieve results."
            )

        logger.info(
            "SearchSession starting run_to_completion session_id=%s prompts=%d",
            self._session_id,
            len(self._prompts),
        )

        # If checkpointing is enabled and the searcher supports step execution,
        # use the public step-wise protocol.
        if (
            self._state_manager is not None
            and (
                self._checkpoint_every_generation
                or self._checkpoint_interval_seconds is not None
            )
            and isinstance(self._searcher, SteppableSearcher)
            and self._searcher.supports_step_execution
        ):
            while not self.is_finished():
                self.run_step()

            result = self._result
            assert result is not None
        else:
            # Delegate directly to searcher
            result = self._searcher.search(self._prompts)
            self._result = result
            self._finished = True

        logger.info(
            "SearchSession completed session_id=%s best_scores=%s",
            self._session_id,
            {k: f"{v:.4f}" for k, v in result.best_scores.items()},
        )

        return result

    def run_step(self) -> StepResult:
        """Execute a single search step.

        Currently only supports evolutionary mode with global_batch scope.

        Returns:
            StepResult with generation info and checkpoint path

        Raises:
            SessionStateError: If search has already finished
            NotImplementedError: For iterative mode or unsupported scopes
        """
        if self._finished:
            raise SessionStateError("Session has already finished.")

        if not isinstance(self._searcher, SteppableSearcher):
            raise NotImplementedError(
                "Step execution only supported for searchers implementing SteppableSearcher"
            )

        searcher = self._searcher
        if not searcher.supports_step_execution:
            raise NotImplementedError(
                "This searcher configuration does not support step execution."
            )

        # Initialize on first step
        if not self._initialized:
            self._step_state = searcher.initialize_step_state(self._prompts)
            self._initialized = True

        assert self._step_state is not None

        current_gen = self._current_generation
        max_generations = int(self._step_state.get("max_steps", searcher.max_steps))

        if current_gen >= max_generations:
            # Finalize and finish
            self._result = searcher.finalize_step_result(
                self._prompts, self._step_state
            )
            self._finished = True
            return StepResult(
                generation=current_gen,
                is_finished=True,
                best_scores=self._step_state["best_scores"],
            )

        # Execute one generation
        searcher.step(self._prompts, self._step_state, current_gen)
        self._current_generation = current_gen + 1

        # Save checkpoint if enabled
        checkpoint_path: Path | None = None
        should_checkpoint = self._should_checkpoint()

        if should_checkpoint and self._state_manager is not None:
            checkpoint_path = self._save_checkpoint(self._dataset_path)

        # Check if we should stop early due to stagnation
        is_finished = (
            self._current_generation >= max_generations
            or self._step_state.get("should_stop", False)
        )

        if is_finished:
            self._result = searcher.finalize_step_result(
                self._prompts, self._step_state
            )
            self._finished = True

        return StepResult(
            generation=self._current_generation,
            is_finished=is_finished,
            best_scores=self._step_state["best_scores"],
            checkpoint_path=checkpoint_path,
        )

    def _should_checkpoint(self) -> bool:
        """Decide whether current step should emit a checkpoint."""
        if self._state_manager is None:
            return False
        if self._checkpoint_every_generation:
            return True
        if self._checkpoint_interval_seconds is None:
            return False
        if self._checkpoint_interval_seconds <= 0:
            return True
        if self._last_checkpoint_time is None:
            return True
        elapsed = (
            datetime.now(timezone.utc) - self._last_checkpoint_time
        ).total_seconds()
        return elapsed >= self._checkpoint_interval_seconds

    def _save_checkpoint(self, dataset_path: Path | None = None) -> Path | None:
        """Save current state as checkpoint.

        Args:
            dataset_path: Optional path to dataset for hash computation

        Returns:
            Path to saved checkpoint, or None when checkpointing is not enabled.

        Raises:
            CheckpointSaveError: If checkpoint serialization or persistence fails.
        """
        if self._state_manager is None or self._step_state is None:
            return None

        try:
            # Serialize RNG state
            rng_state: dict[str, Any] = {}
            if hasattr(self._searcher, "rng"):
                state = self._searcher.rng.getstate()
                rng_state = {
                    "random_state": _serialize_rng_state(state),
                }

            # Serialize scheduler state
            scheduler_state: dict[str, Any] = {}
            if hasattr(self._searcher, "mutation_scheduler"):
                mutation_scheduler = self._searcher.mutation_scheduler
                if hasattr(mutation_scheduler, "get_state"):
                    scheduler_state = mutation_scheduler.get_state()
                else:
                    scheduler_state = mutation_scheduler.get_diagnostics()

            algorithm_state = (
                self._searcher.get_algorithm_state(self._step_state)
                if isinstance(self._searcher, SteppableSearcher)
                else {}
            )

            # Compute dataset hash if path provided
            dataset_hash = "unknown"
            if dataset_path is not None:
                dataset_hash = compute_dataset_hash(dataset_path)

            checkpoint = SearchCheckpoint(
                session_id=self._session_id,
                generation=self._current_generation,
                best_rubrics=self._step_state.get("best_rubrics", {}),
                best_scores=self._step_state.get("best_scores", {}),
                history=self._step_state.get("history", {}),
                scheduler_state=scheduler_state,
                rng_state=rng_state,
                algorithm_state=algorithm_state,
                config_hash=compute_config_hash(_config_to_dict(self._config)),
                dataset_hash=dataset_hash,
            )

            path = self._state_manager.save_checkpoint(checkpoint)
            self._last_checkpoint_time = datetime.now(timezone.utc)

            logger.debug("Checkpoint saved path=%s", path)
            return path

        except (OSError, TypeError, ValueError, CheckpointSaveError) as e:
            message = (
                "Failed to save checkpoint "
                f"session_id={self._session_id} generation={self._current_generation}: {e}"
            )
            logger.error(message)
            raise CheckpointSaveError(message) from e

    def is_finished(self) -> bool:
        """Check if search has completed.

        Returns:
            True if search has finished, False otherwise
        """
        return self._finished

    def get_result(self) -> SearchResult:
        """Get the search result.

        Returns:
            SearchResult from completed search

        Raises:
            SessionStateError: If search has not completed yet
        """
        if not self._finished or self._result is None:
            raise SessionStateError(
                "Search has not completed yet. Call run_to_completion() or run_step() until finished."
            )
        return self._result

    def checkpoint(self) -> Path | None:
        """Manually trigger a checkpoint save.

        Returns:
            Path to saved checkpoint, or None if checkpointing not enabled
        """
        return self._save_checkpoint(self._dataset_path)

    def get_session_info(self) -> dict[str, Any]:
        """Get session metadata for reporting.

        Returns:
            Dictionary with session metadata
        """
        return {
            "session_id": self._session_id,
            "is_resumed": self._is_resumed,
            "resume_source": self._resume_source,
            "resume_semantics": self._resume_semantics,
            "current_generation": self._current_generation,
            "finished": self._finished,
            "checkpoint_enabled": self._state_manager is not None,
            "checkpoint_every_generation": self._checkpoint_every_generation,
            "checkpoint_interval_seconds": self._checkpoint_interval_seconds,
        }


def _config_to_dict(config: RuntimeConfig) -> dict[str, Any]:
    """Convert RuntimeConfig to dictionary for hashing.

    Args:
        config: Runtime configuration

    Returns:
        Dictionary representation
    """
    from dataclasses import asdict, is_dataclass
    from enum import Enum

    def convert(obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        if is_dataclass(obj):
            return convert(asdict(obj))
        if isinstance(obj, dict):
            return {
                str(key): convert(value)
                for key, value in obj.items()
                if key != "api_key"
            }
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        if isinstance(obj, tuple):
            return [convert(item) for item in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    payload = convert(config)
    if not isinstance(payload, dict):
        raise TypeError("RuntimeConfig serialization must produce a dictionary")
    return payload


def _serialize_rng_state(value: Any) -> Any:
    """Convert RNG state tuple into JSON-compatible lists."""
    if isinstance(value, tuple):
        return [_serialize_rng_state(item) for item in value]
    if isinstance(value, list):
        return [_serialize_rng_state(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_rng_state(item) for key, item in value.items()}
    return value


def _deserialize_rng_state(value: Any) -> Any:
    """Convert JSON-compatible RNG payload back into tuple state."""
    if isinstance(value, list):
        return tuple(_deserialize_rng_state(item) for item in value)
    if isinstance(value, dict):
        return {key: _deserialize_rng_state(item) for key, item in value.items()}
    return value


def _restore_rng_state(searcher: Any, rng_state: dict[str, Any]) -> bool:
    """Restore RNG state from checkpoint payload."""
    if not rng_state or not hasattr(searcher, "rng"):
        return True
    try:
        if "random_state" in rng_state:
            searcher.rng.setstate(_deserialize_rng_state(rng_state["random_state"]))
            return True

        # Backward compatibility for Stage 1 checkpoints.
        if "version" in rng_state and "state" in rng_state:
            version = int(rng_state["version"])
            raw_state = rng_state["state"]
            if isinstance(raw_state, list):
                searcher.rng.setstate((version, tuple(raw_state), None))
                return True
        return False
    except Exception as exc:
        logger.warning("Failed to restore RNG state: %s", exc)
        return False


def _restore_scheduler_state(searcher: Any, scheduler_state: dict[str, Any]) -> bool:
    """Restore mutation scheduler state from checkpoint payload."""
    if not scheduler_state or not hasattr(searcher, "mutation_scheduler"):
        return True
    scheduler = searcher.mutation_scheduler
    try:
        if hasattr(scheduler, "set_state"):
            scheduler.set_state(scheduler_state)
            return True

        # Backward-compatible fallback for schedulers exposing generation_count.
        generation = scheduler_state.get("generation")
        if (
            isinstance(generation, int)
            and hasattr(scheduler, "history")
            and hasattr(scheduler.history, "generation_count")
        ):
            scheduler.history.generation_count = generation
            return True
        return False
    except Exception as exc:
        logger.warning("Failed to restore scheduler state: %s", exc)
        return False


def _parse_checkpoint_timestamp(timestamp: str) -> datetime | None:
    """Parse checkpoint timestamp into aware datetime."""
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
