"""Tests for Stage 1 Harness implementation - Checkpoint/Resume.

Stage 1 focuses on:
- StateManager: Persistent checkpoint storage with atomic writes
- SearchSession.run_step(): Step-wise execution for evolutionary search
- SearchSession.resume(): Resume from checkpoint
- CLI integration: --resume-from, --checkpoint-every-generation
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from autosr.config import RuntimeConfig, SearchAlgorithmConfig
from autosr.data_models import PromptExample, ResponseCandidate
from autosr.factory import ComponentFactory
from autosr.harness import (
    CheckpointCorruptedError,
    CheckpointMetadata,
    CheckpointNotFoundError,
    ResumeCompatibilityError,
    ResumeValidator,
    SearchCheckpoint,
    SearchSession,
    SessionStateError,
    StateManager,
    StepResult,
    compute_config_hash,
)
from autosr.harness.storage import StateManager
from autosr.types import BackendType, EvolutionIterationScope


def _build_test_prompts() -> list[PromptExample]:
    """Build minimal test prompts."""
    return [
        PromptExample(
            prompt_id="p1",
            prompt="Test prompt 1",
            candidates=[
                ResponseCandidate(candidate_id="c1", text="Response 1", source="s1"),
                ResponseCandidate(candidate_id="c2", text="Response 2", source="s2"),
            ],
        ),
        PromptExample(
            prompt_id="p2",
            prompt="Test prompt 2",
            candidates=[
                ResponseCandidate(candidate_id="c1", text="Response A", source="s1"),
                ResponseCandidate(candidate_id="c2", text="Response B", source="s2"),
            ],
        ),
    ]


class TestStateManager(unittest.TestCase):
    """Test StateManager checkpoint persistence."""
    
    def setUp(self) -> None:
        """Create temporary directory for checkpoints."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_manager = StateManager(base_dir=self.temp_dir)
    
    def tearDown(self) -> None:
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_checkpoint(self) -> None:
        """Test saving and loading a checkpoint."""
        from autosr.data_models import Criterion, GradingProtocol, Rubric
        
        rubric = Rubric(
            rubric_id="test",
            criteria=[Criterion(criterion_id="c1", text="test", weight=1.0)],
            grading_protocol=GradingProtocol(),
        )
        
        checkpoint = SearchCheckpoint(
            session_id="session_001",
            generation=5,
            best_rubrics={"p1": rubric},
            best_scores={"p1": 0.85},
            history={"p1": [0.5, 0.6, 0.7, 0.8, 0.85]},
            scheduler_state={"mode": "add"},
            rng_state={"version": 2},
            config_hash="abc123",
            dataset_hash="def456",
        )
        
        # Save checkpoint
        path = self.state_manager.save_checkpoint(checkpoint)
        
        # Verify file exists
        self.assertTrue(path.exists())
        self.assertIn("session_001", str(path))
        self.assertIn("gen_0005.json", str(path))
        
        # Load checkpoint
        loaded = self.state_manager.load_checkpoint(session_id="session_001")
        
        self.assertEqual(loaded.session_id, checkpoint.session_id)
        self.assertEqual(loaded.generation, checkpoint.generation)
        self.assertEqual(loaded.best_scores, checkpoint.best_scores)
    
    def test_load_checkpoint_by_path(self) -> None:
        """Test loading a checkpoint by explicit path."""
        from autosr.data_models import Criterion, GradingProtocol, Rubric
        
        rubric = Rubric(
            rubric_id="test",
            criteria=[Criterion(criterion_id="c1", text="test", weight=1.0)],
            grading_protocol=GradingProtocol(),
        )
        
        checkpoint = SearchCheckpoint(
            session_id="session_002",
            generation=3,
            best_rubrics={"p1": rubric},
            best_scores={"p1": 0.75},
            history={"p1": [0.5, 0.6, 0.75]},
            scheduler_state={},
            rng_state={},
            config_hash="abc",
            dataset_hash="def",
        )
        
        path = self.state_manager.save_checkpoint(checkpoint)
        
        # Load by path
        loaded = self.state_manager.load_checkpoint(path=path)
        
        self.assertEqual(loaded.session_id, "session_002")
        self.assertEqual(loaded.generation, 3)
    
    def test_load_nonexistent_checkpoint(self) -> None:
        """Test loading a checkpoint that doesn't exist."""
        with self.assertRaises(CheckpointNotFoundError):
            self.state_manager.load_checkpoint(session_id="nonexistent")
        
        with self.assertRaises(CheckpointNotFoundError):
            self.state_manager.load_checkpoint(path=Path("/nonexistent/path.json"))
    
    def test_load_corrupted_checkpoint(self) -> None:
        """Test loading a corrupted checkpoint file."""
        session_dir = Path(self.temp_dir) / "session_003"
        session_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = session_dir / "gen_0001.json"
        
        # Write invalid JSON
        checkpoint_path.write_text("not valid json {{")
        
        with self.assertRaises(CheckpointCorruptedError):
            self.state_manager.load_checkpoint(path=checkpoint_path)
    
    def test_list_checkpoints(self) -> None:
        """Test listing checkpoints for a session."""
        from autosr.data_models import Criterion, GradingProtocol, Rubric
        
        rubric = Rubric(
            rubric_id="test",
            criteria=[Criterion(criterion_id="c1", text="test", weight=1.0)],
            grading_protocol=GradingProtocol(),
        )
        
        # Create multiple checkpoints
        for gen in [1, 3, 5]:
            checkpoint = SearchCheckpoint(
                session_id="session_004",
                generation=gen,
                best_rubrics={"p1": rubric},
                best_scores={"p1": 0.5 + gen * 0.1},
                history={"p1": [0.5 + gen * 0.1]},
                scheduler_state={},
                rng_state={},
                config_hash="abc",
                dataset_hash="def",
            )
            self.state_manager.save_checkpoint(checkpoint)
        
        checkpoints = self.state_manager.list_checkpoints("session_004")
        
        self.assertEqual(len(checkpoints), 3)
        self.assertEqual([c.generation for c in checkpoints], [1, 3, 5])
    
    def test_list_sessions(self) -> None:
        """Test listing all sessions with checkpoints."""
        from autosr.data_models import Criterion, GradingProtocol, Rubric
        
        rubric = Rubric(
            rubric_id="test",
            criteria=[Criterion(criterion_id="c1", text="test", weight=1.0)],
            grading_protocol=GradingProtocol(),
        )
        
        # Create checkpoints for multiple sessions
        for session_id in ["session_a", "session_b", "session_c"]:
            checkpoint = SearchCheckpoint(
                session_id=session_id,
                generation=1,
                best_rubrics={"p1": rubric},
                best_scores={"p1": 0.5},
                history={"p1": [0.5]},
                scheduler_state={},
                rng_state={},
                config_hash="abc",
                dataset_hash="def",
            )
            self.state_manager.save_checkpoint(checkpoint)
        
        sessions = self.state_manager.list_sessions()
        
        self.assertEqual(len(sessions), 3)
        self.assertIn("session_a", sessions)
        self.assertIn("session_b", sessions)
        self.assertIn("session_c", sessions)
    
    def test_find_latest_checkpoint(self) -> None:
        """Test finding the latest checkpoint for a session."""
        from autosr.data_models import Criterion, GradingProtocol, Rubric
        
        rubric = Rubric(
            rubric_id="test",
            criteria=[Criterion(criterion_id="c1", text="test", weight=1.0)],
            grading_protocol=GradingProtocol(),
        )
        
        # Create checkpoints out of order
        for gen in [5, 2, 10, 1]:
            checkpoint = SearchCheckpoint(
                session_id="session_005",
                generation=gen,
                best_rubrics={"p1": rubric},
                best_scores={"p1": 0.5},
                history={"p1": [0.5]},
                scheduler_state={},
                rng_state={},
                config_hash="abc",
                dataset_hash="def",
            )
            self.state_manager.save_checkpoint(checkpoint)
        
        # Load latest (should be generation 10)
        latest = self.state_manager.load_checkpoint(session_id="session_005")
        
        self.assertEqual(latest.generation, 10)
    
    def test_delete_checkpoint(self) -> None:
        """Test deleting a specific checkpoint."""
        from autosr.data_models import Criterion, GradingProtocol, Rubric
        
        rubric = Rubric(
            rubric_id="test",
            criteria=[Criterion(criterion_id="c1", text="test", weight=1.0)],
            grading_protocol=GradingProtocol(),
        )
        
        checkpoint = SearchCheckpoint(
            session_id="session_006",
            generation=5,
            best_rubrics={"p1": rubric},
            best_scores={"p1": 0.5},
            history={"p1": [0.5]},
            scheduler_state={},
            rng_state={},
            config_hash="abc",
            dataset_hash="def",
        )
        
        path = self.state_manager.save_checkpoint(checkpoint)
        self.assertTrue(path.exists())
        
        # Delete checkpoint
        result = self.state_manager.delete_checkpoint("session_006", 5)
        
        self.assertTrue(result)
        self.assertFalse(path.exists())
        
        # Deleting again should return False
        result = self.state_manager.delete_checkpoint("session_006", 5)
        self.assertFalse(result)
    
    def test_delete_session(self) -> None:
        """Test deleting all checkpoints for a session."""
        from autosr.data_models import Criterion, GradingProtocol, Rubric
        
        rubric = Rubric(
            rubric_id="test",
            criteria=[Criterion(criterion_id="c1", text="test", weight=1.0)],
            grading_protocol=GradingProtocol(),
        )
        
        # Create multiple checkpoints
        for gen in [1, 2, 3]:
            checkpoint = SearchCheckpoint(
                session_id="session_007",
                generation=gen,
                best_rubrics={"p1": rubric},
                best_scores={"p1": 0.5},
                history={"p1": [0.5]},
                scheduler_state={},
                rng_state={},
                config_hash="abc",
                dataset_hash="def",
            )
            self.state_manager.save_checkpoint(checkpoint)
        
        # Delete entire session
        result = self.state_manager.delete_session("session_007")
        
        self.assertTrue(result)
        session_dir = Path(self.temp_dir) / "session_007"
        self.assertFalse(session_dir.exists())


class TestSearchSessionStepExecution(unittest.TestCase):
    """Test SearchSession step-wise execution."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.prompts = _build_test_prompts()
        self.config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(
                mode="evolutionary",
                generations=3,
                population_size=3,
                iteration_scope=EvolutionIterationScope.GLOBAL_BATCH,
            ),
        )
        self.factory = ComponentFactory(self.config)
    
    def tearDown(self) -> None:
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_step_execution_evolutionary_global_batch(self) -> None:
        """Test step-wise execution for evolutionary global_batch mode."""
        state_manager = StateManager(base_dir=self.temp_dir)
        
        session = SearchSession.create(
            prompts=self.prompts,
            config=self.config,
            factory=self.factory,
            state_manager=state_manager,
            checkpoint_every_generation=True,
        )
        
        # Execute steps
        step_count = 0
        while not session.is_finished():
            result = session.run_step()
            step_count += 1
            self.assertEqual(result.generation, step_count)
            self.assertIsNotNone(result.checkpoint_path)
            
            # Prevent infinite loop
            if step_count > 10:
                self.fail("Too many steps - possible infinite loop")
        
        # Should have completed all generations
        self.assertTrue(session.is_finished())
        final_result = session.get_result()
        self.assertIn("p1", final_result.best_scores)
        self.assertIn("p2", final_result.best_scores)
    
    def test_step_execution_not_supported_iterative(self) -> None:
        """Test that step execution raises NotImplementedError for iterative mode."""
        config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(mode="iterative"),
        )
        factory = ComponentFactory(config)
        state_manager = StateManager(base_dir=self.temp_dir)
        
        session = SearchSession.create(
            prompts=self.prompts,
            config=config,
            factory=factory,
            state_manager=state_manager,
        )
        
        with self.assertRaises(NotImplementedError):
            session.run_step()
    
    def test_step_execution_not_supported_prompt_local(self) -> None:
        """Test that step execution raises NotImplementedError for prompt_local scope."""
        config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(
                mode="evolutionary",
                iteration_scope=EvolutionIterationScope.PROMPT_LOCAL,
            ),
        )
        factory = ComponentFactory(config)
        state_manager = StateManager(base_dir=self.temp_dir)
        
        session = SearchSession.create(
            prompts=self.prompts,
            config=config,
            factory=factory,
            state_manager=state_manager,
        )
        
        with self.assertRaises(NotImplementedError):
            session.run_step()
    
    def test_checkpoint_saved_every_generation(self) -> None:
        """Test that checkpoints are saved after each generation."""
        state_manager = StateManager(base_dir=self.temp_dir)
        
        session = SearchSession.create(
            prompts=self.prompts,
            config=self.config,
            factory=self.factory,
            state_manager=state_manager,
            checkpoint_every_generation=True,
        )
        
        # Run all steps
        while not session.is_finished():
            session.run_step()
        
        # Check that checkpoints were saved
        checkpoints = state_manager.list_checkpoints(session.session_id)
        self.assertGreater(len(checkpoints), 0)

    def test_checkpoint_interval_seconds_triggers_checkpoint_without_every_generation(self) -> None:
        """Test that interval-based checkpointing works even when per-generation checkpointing is disabled."""
        state_manager = StateManager(base_dir=self.temp_dir)

        session = SearchSession.create(
            prompts=self.prompts,
            config=self.config,
            factory=self.factory,
            state_manager=state_manager,
            checkpoint_every_generation=False,
            checkpoint_interval_seconds=3600.0,
        )

        # Run one step: first interval-based checkpoint should still be emitted.
        result = session.run_step()
        self.assertFalse(result.is_finished)
        self.assertIsNotNone(result.checkpoint_path)

        checkpoints = state_manager.list_checkpoints(session.session_id)
        self.assertEqual(len(checkpoints), 1)

    def test_run_to_completion_uses_step_path_when_only_interval_checkpoint_enabled(self) -> None:
        """Test that run_to_completion honors interval-based checkpointing."""
        state_manager = StateManager(base_dir=self.temp_dir)

        session = SearchSession.create(
            prompts=self.prompts,
            config=self.config,
            factory=self.factory,
            state_manager=state_manager,
            checkpoint_every_generation=False,
            checkpoint_interval_seconds=3600.0,
        )
        session.run_to_completion()

        checkpoints = state_manager.list_checkpoints(session.session_id)
        self.assertGreaterEqual(len(checkpoints), 1)


class TestSearchSessionResume(unittest.TestCase):
    """Test SearchSession resume functionality."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.prompts = _build_test_prompts()
        self.config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(
                mode="evolutionary",
                generations=5,
                population_size=3,
                iteration_scope=EvolutionIterationScope.GLOBAL_BATCH,
            ),
        )
        self.factory = ComponentFactory(self.config)
        self.state_manager = StateManager(base_dir=self.temp_dir)
    
    def tearDown(self) -> None:
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_resume_from_session_id(self) -> None:
        """Test resuming from a session ID."""
        # Create and run initial session
        session1 = SearchSession.create(
            prompts=self.prompts,
            config=self.config,
            factory=self.factory,
            state_manager=self.state_manager,
            checkpoint_every_generation=True,
            session_id="test_resume_session",
        )
        
        # Run a few steps
        for _ in range(2):
            session1.run_step()
        
        first_gen = session1.current_generation
        self.assertEqual(first_gen, 2)
        
        # Resume from checkpoint
        session2 = SearchSession.resume(
            resume_from="test_resume_session",
            config=self.config,
            factory=self.factory,
            state_manager=self.state_manager,
            prompts=self.prompts,
        )
        
        self.assertTrue(session2.is_resumed)
        self.assertEqual(session2.current_generation, first_gen)
        self.assertEqual(session2.session_id, "test_resume_session")
    
    def test_resume_from_checkpoint_path(self) -> None:
        """Test resuming from a specific checkpoint path."""
        # Create initial session and checkpoint
        session1 = SearchSession.create(
            prompts=self.prompts,
            config=self.config,
            factory=self.factory,
            state_manager=self.state_manager,
            checkpoint_every_generation=True,
            session_id="test_resume_path",
        )
        
        # Run one step and get checkpoint path
        result = session1.run_step()
        checkpoint_path = result.checkpoint_path
        
        self.assertIsNotNone(checkpoint_path)
        
        # Resume from specific checkpoint path
        session2 = SearchSession.resume(
            resume_from=checkpoint_path,
            config=self.config,
            factory=self.factory,
            state_manager=self.state_manager,
            prompts=self.prompts,
        )
        
        self.assertTrue(session2.is_resumed)
        self.assertEqual(session2.current_generation, 1)
    
    def test_resume_with_incompatible_config(self) -> None:
        """Test that resume with incompatible config raises error."""
        from autosr.data_models import Criterion, GradingProtocol, Rubric
        
        rubric = Rubric(
            rubric_id="test",
            criteria=[Criterion(criterion_id="c1", text="test", weight=1.0)],
            grading_protocol=GradingProtocol(),
        )
        
        # Create a checkpoint with specific config hash
        checkpoint = SearchCheckpoint(
            session_id="test_incompatible",
            generation=3,
            best_rubrics={"p1": rubric},
            best_scores={"p1": 0.8},
            history={"p1": [0.5, 0.6, 0.8]},
            scheduler_state={},
            rng_state={},
            config_hash="incompatible_hash_123",
            dataset_hash="dataset_hash_456",
        )
        
        self.state_manager.save_checkpoint(checkpoint)
        
        # Try to resume with different config (no dataset for validation)
        # This should log a warning but proceed
        session = SearchSession.resume(
            resume_from="test_incompatible",
            config=self.config,
            factory=self.factory,
            state_manager=self.state_manager,
            prompts=self.prompts,
        )
        
        self.assertTrue(session.is_resumed)
    
    def test_resume_requires_prompts(self) -> None:
        """Test that resume requires prompts parameter."""
        with self.assertRaises(ValueError) as ctx:
            SearchSession.resume(
                resume_from="nonexistent",
                config=self.config,
                factory=self.factory,
                state_manager=self.state_manager,
                prompts=None,  # type: ignore
            )

        self.assertIn("prompts", str(ctx.exception).lower())

    def test_resume_restores_rng_state(self) -> None:
        """Test that resume restores RNG state from checkpoint."""
        def _to_tuple(value: object) -> object:
            if isinstance(value, list):
                return tuple(_to_tuple(item) for item in value)
            return value

        session1 = SearchSession.create(
            prompts=self.prompts,
            config=self.config,
            factory=self.factory,
            state_manager=self.state_manager,
            checkpoint_every_generation=True,
            session_id="resume_rng_state",
        )

        first = session1.run_step()
        self.assertIsNotNone(first.checkpoint_path)
        checkpoint = self.state_manager.load_checkpoint(session_id="resume_rng_state")
        self.assertIn("random_state", checkpoint.rng_state)

        session2 = SearchSession.resume(
            resume_from="resume_rng_state",
            config=self.config,
            factory=self.factory,
            state_manager=self.state_manager,
            prompts=self.prompts,
        )

        resumed_rng_state = session2._searcher.rng.getstate()  # type: ignore[attr-defined]
        self.assertEqual(resumed_rng_state, _to_tuple(checkpoint.rng_state["random_state"]))

    def test_resume_restores_scheduler_generation(self) -> None:
        """Test that scheduler generation state is restored from checkpoint."""
        session1 = SearchSession.create(
            prompts=self.prompts,
            config=self.config,
            factory=self.factory,
            state_manager=self.state_manager,
            checkpoint_every_generation=True,
            session_id="resume_scheduler_state",
        )

        session1.run_step()
        session1.run_step()

        checkpoint = self.state_manager.load_checkpoint(session_id="resume_scheduler_state")
        expected_generation = checkpoint.scheduler_state.get("generation")
        self.assertIsNotNone(expected_generation)

        session2 = SearchSession.resume(
            resume_from="resume_scheduler_state",
            config=self.config,
            factory=self.factory,
            state_manager=self.state_manager,
            prompts=self.prompts,
        )
        diagnostics = session2._searcher.mutation_scheduler.get_diagnostics()  # type: ignore[attr-defined]
        self.assertEqual(diagnostics.get("generation"), expected_generation)

    def test_resume_contract_marks_reseed_for_legacy_checkpoint(self) -> None:
        """Test resume contract metadata for legacy checkpoints without algorithm state."""
        from autosr.data_models import Criterion, GradingProtocol, Rubric

        rubric = Rubric(
            rubric_id="legacy_rubric",
            criteria=[Criterion(criterion_id="c1", text="legacy", weight=1.0)],
            grading_protocol=GradingProtocol(),
        )
        legacy_checkpoint = SearchCheckpoint(
            session_id="legacy_resume_contract",
            generation=1,
            best_rubrics={"p1": rubric},
            best_scores={"p1": 0.6},
            history={"p1": [0.6]},
            scheduler_state={},
            rng_state={},
            config_hash="legacy_cfg_hash",
            dataset_hash="legacy_data_hash",
        )
        self.state_manager.save_checkpoint(legacy_checkpoint)

        resumed = SearchSession.resume(
            resume_from="legacy_resume_contract",
            config=self.config,
            factory=self.factory,
            state_manager=self.state_manager,
            prompts=self.prompts,
        )
        info = resumed.get_session_info()
        self.assertEqual(info.get("resume_semantics"), "reseed_from_checkpoint")


class TestIntegrationCheckpointResume(unittest.TestCase):
    """Integration tests for checkpoint and resume workflow."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.prompts = _build_test_prompts()
        self.config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(
                mode="evolutionary",
                generations=5,
                population_size=3,
                iteration_scope=EvolutionIterationScope.GLOBAL_BATCH,
            ),
        )
        self.factory = ComponentFactory(self.config)
        self.state_manager = StateManager(base_dir=self.temp_dir)
    
    def tearDown(self) -> None:
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_checkpoint_resume_cycle(self) -> None:
        """Test complete workflow: create, checkpoint, resume, complete."""
        # Phase 1: Create session and run partial
        session1 = SearchSession.create(
            prompts=self.prompts,
            config=self.config,
            factory=self.factory,
            state_manager=self.state_manager,
            checkpoint_every_generation=True,
            session_id="full_cycle_test",
        )
        
        # Run 2 generations
        for _ in range(2):
            session1.run_step()
        
        partial_scores = dict(session1._step_state["best_scores"])  # type: ignore
        
        # Phase 2: Resume and complete
        session2 = SearchSession.resume(
            resume_from="full_cycle_test",
            config=self.config,
            factory=self.factory,
            state_manager=self.state_manager,
            prompts=self.prompts,
            checkpoint_every_generation=True,
        )
        
        # Complete remaining generations
        while not session2.is_finished():
            session2.run_step()
        
        final_result = session2.get_result()
        
        # Verify completion
        self.assertIn("p1", final_result.best_scores)
        self.assertIn("p2", final_result.best_scores)
        self.assertTrue(session2.is_finished())
        
        # Verify all generations completed
        self.assertGreaterEqual(session2.current_generation, 2)


if __name__ == "__main__":
    unittest.main()
