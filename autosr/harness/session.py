"""Search session lifecycle management for harness-based execution.

This module provides the SearchSession wrapper that adds session management
capabilities to existing search algorithms without modifying their core logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..data_models import PromptExample, Rubric
from ..search.config import SearchResult
from ..types import EvolutionIterationScope
from .state import ResumeCompatibilityError, SearchCheckpoint, compute_config_hash, compute_dataset_hash

if TYPE_CHECKING:
    from ..config import RuntimeConfig
    from ..factory import ComponentFactory
    from ..search import EvolutionaryRTDSearcher, IterativeRTDSearcher
    from .storage import StateManager

logger = logging.getLogger("autosr.harness")


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
        searcher: IterativeRTDSearcher | EvolutionaryRTDSearcher,
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
        self._last_checkpoint_time: datetime | None = None
        
        # For step-wise execution (Stage 1)
        self._step_state: dict[str, Any] | None = None
        self._initialized = False
        
        logger.info(
            "SearchSession initialized session_id=%s mode=%s checkpoint_enabled=%s",
            session_id,
            config.search.mode.value if hasattr(config.search.mode, 'value') else str(config.search.mode),
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
            if checkpoint.dataset_hash != "unknown" and checkpoint.dataset_hash != current_dataset_hash:
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
        
        # Restore RNG state if available
        if checkpoint.rng_state and hasattr(searcher, 'rng'):
            try:
                import random
                searcher.rng.setstate(tuple(checkpoint.rng_state.get('state', ())))
            except Exception as e:
                logger.warning("Failed to restore RNG state: %s", e)
        
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
        
        # Initialize step state for continued execution
        # For evolutionary mode, we need to re-initialize populations
        if (hasattr(searcher, 'config') and 
            hasattr(searcher.config, 'iteration_scope')):
            # Re-initialize global state from checkpoint
            population, history, best_rubrics, best_scores, initial_margins = searcher._init_global_state(prompts)
            
            # Override with checkpoint values
            best_rubrics.update(checkpoint.best_rubrics)
            best_scores.update(checkpoint.best_scores)
            for pid, scores in checkpoint.history.items():
                if pid in history:
                    history[pid] = list(scores)
            
            session._step_state = {
                "population": population,
                "history": history,
                "best_rubrics": best_rubrics,
                "best_scores": best_scores,
                "initial_margins": initial_margins,
                "stale_rounds": 0,
                "should_stop": False,
            }
            session._initialized = True
        else:
            # Fallback for other modes
            session._step_state = {
                "best_rubrics": checkpoint.best_rubrics,
                "best_scores": checkpoint.best_scores,
                "history": checkpoint.history,
            }
            session._initialized = False
        
        logger.info(
            "SearchSession resumed session_id=%s generation=%d source=%s",
            checkpoint.session_id,
            checkpoint.generation,
            resume_from,
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
            raise SessionStateError("Session has already finished. Use get_result() to retrieve results.")
        
        logger.info(
            "SearchSession starting run_to_completion session_id=%s prompts=%d",
            self._session_id,
            len(self._prompts),
        )
        
        # If checkpointing is enabled and we're using evolutionary mode,
        # use step-wise execution
        if (self._state_manager is not None and 
            self._checkpoint_every_generation and
            hasattr(self._searcher, 'config') and
            hasattr(self._searcher.config, 'iteration_scope')):
            
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
        
        # Only support evolutionary mode for now
        if not hasattr(self._searcher, 'config') or not hasattr(self._searcher.config, 'iteration_scope'):
            raise NotImplementedError(
                "Step execution only supported for evolutionary search mode"
            )
        
        searcher: EvolutionaryRTDSearcher = self._searcher  # type: ignore
        config = searcher.config
        
        if config.iteration_scope is EvolutionIterationScope.PROMPT_LOCAL:
            raise NotImplementedError(
                "Step execution for prompt_local scope not yet implemented. "
                "Use global_batch scope for checkpointable search."
            )
        
        # Initialize on first step
        if not self._initialized:
            self._init_global_step_state()
        
        assert self._step_state is not None
        
        current_gen = self._current_generation
        max_generations = config.generations
        
        if current_gen >= max_generations:
            # Finalize and finish
            self._finalize_global_search()
            return StepResult(
                generation=current_gen,
                is_finished=True,
                best_scores=self._step_state["best_scores"],
            )
        
        # Execute one generation
        self._execute_global_generation(current_gen)
        self._current_generation = current_gen + 1
        
        # Save checkpoint if enabled
        checkpoint_path: Path | None = None
        should_checkpoint = self._checkpoint_every_generation
        
        if should_checkpoint and self._state_manager is not None:
            checkpoint_path = self._save_checkpoint(self._dataset_path)
        
        # Check if we should stop early due to stagnation
        is_finished = (
            self._current_generation >= max_generations or
            self._step_state.get("should_stop", False)
        )
        
        if is_finished:
            self._finalize_global_search()
        
        return StepResult(
            generation=self._current_generation,
            is_finished=is_finished,
            best_scores=self._step_state["best_scores"],
            checkpoint_path=checkpoint_path,
        )
    
    def _init_global_step_state(self) -> None:
        """Initialize step-wise execution state for global_batch mode."""
        searcher: EvolutionaryRTDSearcher = self._searcher  # type: ignore
        
        population, history, best_rubrics, best_scores, initial_margins = searcher._init_global_state(self._prompts)
        
        self._step_state = {
            "population": population,
            "history": history,
            "best_rubrics": best_rubrics,
            "best_scores": best_scores,
            "initial_margins": initial_margins,
            "stale_rounds": 0,
            "should_stop": False,
        }
        self._initialized = True
        
        logger.debug("Global step state initialized session_id=%s", self._session_id)
    
    def _execute_global_generation(self, generation: int) -> None:
        """Execute a single generation for global_batch mode."""
        from ..evaluator import ObjectiveBreakdown
        
        searcher: EvolutionaryRTDSearcher = self._searcher  # type: ignore
        assert self._step_state is not None
        
        population = self._step_state["population"]
        history = self._step_state["history"]
        best_rubrics = self._step_state["best_rubrics"]
        best_scores = self._step_state["best_scores"]
        initial_margins = self._step_state["initial_margins"]
        
        # Score population
        scored_population: dict[str, list[tuple[Rubric, ObjectiveBreakdown]]] = {}
        for item in self._prompts:
            scored_population[item.prompt_id] = searcher._score_population(item, population[item.prompt_id])
        
        # Log progress
        searcher._log_generation_progress(generation, scored_population, initial_margins)
        
        # Compute diversity
        diversity_scores: dict[str, float] = {}
        for item in self._prompts:
            diversity = searcher.diversity_metric.compute(
                population[item.prompt_id],
                rng=searcher.rng,
            )
            diversity_scores[item.prompt_id] = diversity
            searcher.diversity_history.append(diversity)
        
        # Update bests
        generation_improved = searcher._update_generation_bests(
            scored_population=scored_population,
            best_rubrics=best_rubrics,
            best_scores=best_scores,
            history=history,
        )
        
        # Handle stagnation
        stale_rounds = self._step_state["stale_rounds"]
        stale_rounds, should_stop = searcher._handle_stagnation(generation_improved, stale_rounds)
        self._step_state["stale_rounds"] = stale_rounds
        self._step_state["should_stop"] = should_stop
        
        if should_stop:
            logger.info(
                "stopping early at generation=%d stale_rounds=%d threshold=%d",
                generation + 1,
                stale_rounds,
                searcher.config.stagnation_generations,
            )
            return
        
        # Evolve selected prompts
        hard_prompt_ids = searcher._select_hard_prompts(self._prompts, population, scored_population)
        logger.info(
            "generation=%d selected_hard_prompts=%s",
            generation + 1,
            ",".join(sorted(hard_prompt_ids)) if hard_prompt_ids else "<none>",
        )
        
        searcher._evolve_selected_prompts(
            prompts=self._prompts,
            hard_prompt_ids=hard_prompt_ids,
            scored_population=scored_population,
            population=population,
            diversity_scores=diversity_scores,
            generation=generation,
            scheduler=searcher.mutation_scheduler,
        )
        
        searcher.mutation_scheduler.next_generation()
    
    def _finalize_global_search(self) -> None:
        """Finalize search and create result."""
        searcher: EvolutionaryRTDSearcher = self._searcher  # type: ignore
        assert self._step_state is not None
        
        population = self._step_state["population"]
        history = self._step_state["history"]
        best_rubrics = self._step_state["best_rubrics"]
        best_scores = self._step_state["best_scores"]
        initial_margins = self._step_state["initial_margins"]
        
        searcher._finalize_best_from_population(
            prompts=self._prompts,
            population=population,
            best_rubrics=best_rubrics,
            best_scores=best_scores,
        )
        
        margin_improvement = searcher._collect_margin_improvement(
            prompts=self._prompts,
            best_rubrics=best_rubrics,
            initial_margins=initial_margins,
        )
        
        diagnostics = {
            "mode": "evolutionary",
            "iteration_scope": searcher.config.iteration_scope.value,
            "selection_strategy": searcher.config.selection_strategy.name,
            "adaptive_mutation": searcher.config.adaptive_mutation.name,
            "mutation_diagnostics": searcher.mutation_scheduler.get_diagnostics(),
            "avg_diversity": sum(searcher.diversity_history) / len(searcher.diversity_history)
            if searcher.diversity_history else 0.0,
            "margin_improvement": margin_improvement,
        }
        
        self._result = SearchResult(
            best_rubrics=best_rubrics,
            best_scores=best_scores,
            history=history,
            diagnostics=diagnostics,
        )
        self._finished = True
        
        logger.info("Search finalized session_id=%s", self._session_id)
    
    def _save_checkpoint(self, dataset_path: Path | None = None) -> Path | None:
        """Save current state as checkpoint.
        
        Args:
            dataset_path: Optional path to dataset for hash computation
            
        Returns:
            Path to saved checkpoint, or None if save failed
        """
        if self._state_manager is None or self._step_state is None:
            return None
        
        try:
            # Serialize RNG state
            rng_state: dict[str, Any] = {}
            if hasattr(self._searcher, 'rng'):
                import random
                state = self._searcher.rng.getstate()
                rng_state = {
                    "version": state[0],
                    "state": list(state[1]) if isinstance(state[1], tuple) else state[1],
                }
            
            # Serialize scheduler state
            scheduler_state: dict[str, Any] = {}
            if hasattr(self._searcher, 'mutation_scheduler'):
                scheduler_state = self._searcher.mutation_scheduler.get_diagnostics()
            
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
                config_hash=compute_config_hash(_config_to_dict(self._config)),
                dataset_hash=dataset_hash,
            )
            
            path = self._state_manager.save_checkpoint(checkpoint)
            self._last_checkpoint_time = datetime.now(timezone.utc)
            
            logger.debug("Checkpoint saved path=%s", path)
            return path
            
        except Exception as e:
            logger.error("Failed to save checkpoint: %s", e)
            return None
    
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
            raise SessionStateError("Search has not completed yet. Call run_to_completion() or run_step() until finished.")
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
            return asdict(obj)
        return obj
    
    return {
        "backend": str(config.backend),
        "search": asdict(config.search),
        "objective": asdict(config.objective),
        "initializer": asdict(config.initializer),
        "extraction": asdict(config.extraction),
        "candidate_extraction": asdict(config.candidate_extraction),
        "verifier": asdict(config.verifier),
        "llm": {
            "base_url": config.llm.base_url,
            "timeout": config.llm.timeout,
            "max_retries": config.llm.max_retries,
            "temperature": config.llm.temperature,
            "default_model": config.llm.default_model,
        },
    }
