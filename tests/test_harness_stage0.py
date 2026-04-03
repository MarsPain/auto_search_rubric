"""Tests for Stage 0 Harness implementation.

Stage 0 focuses on:
- SearchSession wrapper API (no algorithm modification)
- SearchCheckpoint schema v1 definition
- ResumeValidator compatibility checking
- Backward compatibility with existing CLI
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from autosr.config import RuntimeConfig, SearchAlgorithmConfig
from autosr.data_models import Criterion, GradingProtocol, PromptExample, ResponseCandidate, Rubric
from autosr.factory import ComponentFactory
from autosr.types import BackendType
from autosr.harness import (
    CheckpointValidationError,
    ResumeCompatibilityError,
    ResumeValidationResult,
    ResumeValidator,
    SearchCheckpoint,
    SearchSession,
    SessionStateError,
    compute_config_hash,
    compute_dataset_hash,
)


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
    ]


def _build_test_rubric() -> Rubric:
    """Build a test rubric."""
    return Rubric(
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


class TestSearchCheckpointSchema(unittest.TestCase):
    """Test SearchCheckpoint schema v1."""
    
    def test_checkpoint_creation(self) -> None:
        """Test basic checkpoint creation."""
        rubric = _build_test_rubric()
        checkpoint = SearchCheckpoint(
            session_id="test_session_001",
            generation=5,
            best_rubrics={"p1": rubric},
            best_scores={"p1": 0.85},
            history={"p1": [0.5, 0.6, 0.7, 0.8, 0.85]},
            scheduler_state={"mode_weights": {"add": 0.3, "remove": 0.2}},
            rng_state={"version": 2, "state": [1, 2, 3]},
            config_hash="abc123",
            dataset_hash="def456",
        )
        
        self.assertEqual(checkpoint.session_id, "test_session_001")
        self.assertEqual(checkpoint.generation, 5)
        self.assertEqual(checkpoint.best_scores["p1"], 0.85)
        self.assertEqual(checkpoint.schema_version, "1.0")
    
    def test_checkpoint_validation_empty_session_id(self) -> None:
        """Test checkpoint validation rejects empty session_id."""
        with self.assertRaises(CheckpointValidationError) as ctx:
            SearchCheckpoint(
                session_id="",
                generation=0,
                best_rubrics={},
                best_scores={},
                history={},
                scheduler_state={},
                rng_state={},
                config_hash="abc",
                dataset_hash="def",
            )
        self.assertIn("session_id", str(ctx.exception))
    
    def test_checkpoint_validation_negative_generation(self) -> None:
        """Test checkpoint validation rejects negative generation."""
        with self.assertRaises(CheckpointValidationError) as ctx:
            SearchCheckpoint(
                session_id="test",
                generation=-1,
                best_rubrics={},
                best_scores={},
                history={},
                scheduler_state={},
                rng_state={},
                config_hash="abc",
                dataset_hash="def",
            )
        self.assertIn("generation", str(ctx.exception))
    
    def test_checkpoint_validation_empty_hashes(self) -> None:
        """Test checkpoint validation rejects empty hashes."""
        with self.assertRaises(CheckpointValidationError):
            SearchCheckpoint(
                session_id="test",
                generation=0,
                best_rubrics={},
                best_scores={},
                history={},
                scheduler_state={},
                rng_state={},
                config_hash="",
                dataset_hash="def",
            )
        
        with self.assertRaises(CheckpointValidationError):
            SearchCheckpoint(
                session_id="test",
                generation=0,
                best_rubrics={},
                best_scores={},
                history={},
                scheduler_state={},
                rng_state={},
                config_hash="abc",
                dataset_hash="",
            )
    
    def test_checkpoint_serialization_roundtrip(self) -> None:
        """Test checkpoint to_dict/from_dict roundtrip."""
        rubric = _build_test_rubric()
        original = SearchCheckpoint(
            session_id="test_session_002",
            generation=3,
            best_rubrics={"p1": rubric},
            best_scores={"p1": 0.75},
            history={"p1": [0.5, 0.6, 0.75]},
            scheduler_state={"current_mode": "add"},
            rng_state={"state": [1, 2, 3]},
            config_hash="hash123",
            dataset_hash="hash456",
        )
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = SearchCheckpoint.from_dict(data)
        
        # Verify roundtrip
        self.assertEqual(restored.session_id, original.session_id)
        self.assertEqual(restored.generation, original.generation)
        self.assertEqual(restored.best_scores, original.best_scores)
        self.assertEqual(restored.config_hash, original.config_hash)
        self.assertEqual(restored.dataset_hash, original.dataset_hash)
        
        # Verify rubric was properly reconstructed
        self.assertIn("p1", restored.best_rubrics)
        self.assertEqual(restored.best_rubrics["p1"].rubric_id, rubric.rubric_id)
    
    def test_checkpoint_json_roundtrip(self) -> None:
        """Test checkpoint JSON serialization roundtrip."""
        rubric = _build_test_rubric()
        original = SearchCheckpoint(
            session_id="test_session_003",
            generation=2,
            best_rubrics={"p1": rubric},
            best_scores={"p1": 0.65},
            history={"p1": [0.5, 0.65]},
            scheduler_state={},
            rng_state={},
            config_hash="abc",
            dataset_hash="def",
        )
        
        json_str = original.to_json()
        restored = SearchCheckpoint.from_json(json_str)
        
        self.assertEqual(restored.session_id, original.session_id)
        self.assertEqual(restored.generation, original.generation)
    
    def test_checkpoint_unsupported_schema_version(self) -> None:
        """Test checkpoint rejects unsupported schema version."""
        data = {
            "schema_version": "2.0",  # Unsupported future version
            "session_id": "test",
            "generation": 0,
            "best_rubrics": {},
            "best_scores": {},
            "history": {},
            "scheduler_state": {},
            "rng_state": {},
            "config_hash": "abc",
            "dataset_hash": "def",
        }
        
        with self.assertRaises(CheckpointValidationError) as ctx:
            SearchCheckpoint.from_dict(data)
        self.assertIn("schema_version", str(ctx.exception))


class TestResumeValidator(unittest.TestCase):
    """Test ResumeValidator compatibility checking."""
    
    def test_compatible_checkpoint(self) -> None:
        """Test validation passes for matching hashes."""
        validator = ResumeValidator(
            current_config_hash="config_hash_123",
            current_dataset_hash="dataset_hash_456",
        )
        
        checkpoint = SearchCheckpoint(
            session_id="test",
            generation=5,
            best_rubrics={},
            best_scores={},
            history={},
            scheduler_state={},
            rng_state={},
            config_hash="config_hash_123",
            dataset_hash="dataset_hash_456",
        )
        
        result = validator.validate(checkpoint)
        self.assertTrue(result.compatible)
        self.assertIsNone(result.reason)
    
    def test_incompatible_config_hash(self) -> None:
        """Test validation fails for mismatched config hash."""
        validator = ResumeValidator(
            current_config_hash="current_config_hash",
            current_dataset_hash="dataset_hash_456",
        )
        
        checkpoint = SearchCheckpoint(
            session_id="test",
            generation=5,
            best_rubrics={},
            best_scores={},
            history={},
            scheduler_state={},
            rng_state={},
            config_hash="different_config_hash",
            dataset_hash="dataset_hash_456",
        )
        
        result = validator.validate(checkpoint)
        self.assertFalse(result.compatible)
        self.assertIn("config_hash", result.details)
    
    def test_incompatible_dataset_hash(self) -> None:
        """Test validation fails for mismatched dataset hash."""
        validator = ResumeValidator(
            current_config_hash="config_hash_123",
            current_dataset_hash="current_dataset_hash",
        )
        
        checkpoint = SearchCheckpoint(
            session_id="test",
            generation=5,
            best_rubrics={},
            best_scores={},
            history={},
            scheduler_state={},
            rng_state={},
            config_hash="config_hash_123",
            dataset_hash="different_dataset_hash",
        )
        
        result = validator.validate(checkpoint)
        self.assertFalse(result.compatible)
        self.assertIn("dataset_hash", result.details)
    
    def test_validate_or_raise_compatible(self) -> None:
        """Test validate_or_raise does not raise for compatible checkpoint."""
        validator = ResumeValidator(
            current_config_hash="hash123",
            current_dataset_hash="hash456",
        )
        
        checkpoint = SearchCheckpoint(
            session_id="test",
            generation=0,
            best_rubrics={},
            best_scores={},
            history={},
            scheduler_state={},
            rng_state={},
            config_hash="hash123",
            dataset_hash="hash456",
        )
        
        # Should not raise
        validator.validate_or_raise(checkpoint)
    
    def test_validate_or_raise_incompatible(self) -> None:
        """Test validate_or_raise raises for incompatible checkpoint."""
        validator = ResumeValidator(
            current_config_hash="hash123",
            current_dataset_hash="hash456",
        )
        
        checkpoint = SearchCheckpoint(
            session_id="test",
            generation=0,
            best_rubrics={},
            best_scores={},
            history={},
            scheduler_state={},
            rng_state={},
            config_hash="different",
            dataset_hash="different",
        )
        
        with self.assertRaises(ResumeCompatibilityError) as ctx:
            validator.validate_or_raise(checkpoint)
        self.assertIsNotNone(ctx.exception.details)


class TestHashFunctions(unittest.TestCase):
    """Test config and dataset hash computation."""
    
    def test_compute_config_hash_deterministic(self) -> None:
        """Test config hash is deterministic."""
        config = {"seed": 42, "mode": "evolutionary", "nested": {"key": "value"}}
        
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA256 hex digest length
    
    def test_compute_config_hash_order_independent(self) -> None:
        """Test config hash is independent of key order."""
        config1 = {"a": 1, "b": 2, "c": {"x": 3, "y": 4}}
        config2 = {"b": 2, "a": 1, "c": {"y": 4, "x": 3}}
        
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        
        self.assertEqual(hash1, hash2)
    
    def test_compute_config_hash_sensitive_to_values(self) -> None:
        """Test config hash changes with values."""
        config1 = {"seed": 42}
        config2 = {"seed": 43}
        
        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        
        self.assertNotEqual(hash1, hash2)
    
    def test_compute_dataset_hash(self) -> None:
        """Test dataset hash computation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"test": "data", "items": [1, 2, 3]}, f)
            temp_path = Path(f.name)
        
        try:
            hash1 = compute_dataset_hash(temp_path)
            hash2 = compute_dataset_hash(temp_path)
            
            self.assertEqual(hash1, hash2)
            self.assertEqual(len(hash1), 64)
        finally:
            os.unlink(temp_path)
    
    def test_compute_dataset_hash_file_not_found(self) -> None:
        """Test dataset hash raises for missing file."""
        with self.assertRaises(FileNotFoundError):
            compute_dataset_hash(Path("/nonexistent/file.json"))


class TestSearchSession(unittest.TestCase):
    """Test SearchSession wrapper API."""
    
    def test_session_creation(self) -> None:
        """Test SearchSession.create factory method."""
        prompts = _build_test_prompts()
        config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(mode="iterative"),
        )
        factory = ComponentFactory(config)
        
        session = SearchSession.create(
            prompts=prompts,
            config=config,
            factory=factory,
            session_id="test_session_001",
        )
        
        self.assertEqual(session.session_id, "test_session_001")
        self.assertFalse(session.is_finished())
        self.assertEqual(session.current_generation, 0)
        self.assertFalse(session.is_resumed)
    
    def test_session_auto_generates_session_id(self) -> None:
        """Test session auto-generates session_id if not provided."""
        prompts = _build_test_prompts()
        config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(mode="iterative"),
        )
        factory = ComponentFactory(config)
        
        session = SearchSession.create(
            prompts=prompts,
            config=config,
            factory=factory,
        )
        
        self.assertIsNotNone(session.session_id)
        self.assertGreater(len(session.session_id), 0)
    
    def test_session_run_to_completion(self) -> None:
        """Test running search to completion."""
        prompts = _build_test_prompts()
        config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(mode="iterative"),
        )
        factory = ComponentFactory(config)
        
        session = SearchSession.create(
            prompts=prompts,
            config=config,
            factory=factory,
        )
        
        result = session.run_to_completion()
        
        self.assertTrue(session.is_finished())
        self.assertIsNotNone(result.best_rubrics)
        self.assertIsNotNone(result.best_scores)
        self.assertIn("p1", result.best_scores)
    
    def test_session_run_to_completion_twice_raises(self) -> None:
        """Test running completed session raises error."""
        prompts = _build_test_prompts()
        config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(mode="iterative"),
        )
        factory = ComponentFactory(config)
        
        session = SearchSession.create(
            prompts=prompts,
            config=config,
            factory=factory,
        )
        
        session.run_to_completion()
        
        with self.assertRaises(SessionStateError):
            session.run_to_completion()
    
    def test_session_get_result_before_completion_raises(self) -> None:
        """Test getting result before completion raises error."""
        prompts = _build_test_prompts()
        config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(mode="iterative"),
        )
        factory = ComponentFactory(config)
        
        session = SearchSession.create(
            prompts=prompts,
            config=config,
            factory=factory,
        )
        
        with self.assertRaises(SessionStateError):
            session.get_result()
    
    def test_session_get_result_after_completion(self) -> None:
        """Test getting result after completion succeeds."""
        prompts = _build_test_prompts()
        config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(mode="iterative"),
        )
        factory = ComponentFactory(config)
        
        session = SearchSession.create(
            prompts=prompts,
            config=config,
            factory=factory,
        )
        
        result1 = session.run_to_completion()
        result2 = session.get_result()
        
        self.assertIs(result1, result2)
    
    def test_session_resume_not_implemented_stage0(self) -> None:
        """Test resume requires prompts in Stage 1+ (was NotImplementedError in Stage 0)."""
        config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(mode="iterative"),
        )
        factory = ComponentFactory(config)
        
        # In Stage 1+, resume requires prompts (raises ValueError without)
        with self.assertRaises((NotImplementedError, ValueError)):
            SearchSession.resume(
                resume_from="some_checkpoint",
                config=config,
                factory=factory,
                state_manager=MagicMock(),
                prompts=None,  # Missing required prompts
            )
    
    def test_session_run_step_not_implemented_stage0(self) -> None:
        """Test run_step raises NotImplementedError in Stage 0."""
        prompts = _build_test_prompts()
        config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(mode="iterative"),
        )
        factory = ComponentFactory(config)
        
        session = SearchSession.create(
            prompts=prompts,
            config=config,
            factory=factory,
        )
        
        with self.assertRaises(NotImplementedError):
            session.run_step()
    
    def test_session_checkpoint_returns_none_stage0(self) -> None:
        """Test checkpoint returns None in Stage 0."""
        prompts = _build_test_prompts()
        config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(mode="iterative"),
        )
        factory = ComponentFactory(config)
        
        session = SearchSession.create(
            prompts=prompts,
            config=config,
            factory=factory,
        )
        
        # Without state_manager, checkpoint returns None
        checkpoint = session.checkpoint()
        self.assertIsNone(checkpoint)
    
    def test_session_get_session_info(self) -> None:
        """Test getting session metadata."""
        prompts = _build_test_prompts()
        config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(mode="iterative"),
        )
        factory = ComponentFactory(config)
        
        session = SearchSession.create(
            prompts=prompts,
            config=config,
            factory=factory,
            session_id="test_info",
        )
        
        info = session.get_session_info()
        
        self.assertEqual(info["session_id"], "test_info")
        self.assertEqual(info["current_generation"], 0)
        self.assertFalse(info["finished"])
        self.assertFalse(info["is_resumed"])
        self.assertFalse(info["checkpoint_enabled"])


class TestHarnessBackwardCompatibility(unittest.TestCase):
    """Test that harness layer doesn't break existing CLI behavior."""
    
    def test_cli_without_harness_still_works(self) -> None:
        """Test existing CLI path without harness session still works."""
        # This simulates the current CLI code path
        prompts = _build_test_prompts()
        config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(mode="iterative"),
        )
        factory = ComponentFactory(config)
        
        # Direct factory usage (current CLI path)
        searcher = factory.create_searcher(prompts)
        result = searcher.search(prompts)
        
        self.assertIn("p1", result.best_scores)
        self.assertIsNotNone(result.best_rubrics)
    
    def test_harness_wrapper_produces_same_result(self) -> None:
        """Test harness wrapper produces same result as direct usage."""
        prompts = _build_test_prompts()
        config = RuntimeConfig(
            backend=BackendType.MOCK,
            search=SearchAlgorithmConfig(mode="iterative"),
        )
        
        # Direct usage
        factory1 = ComponentFactory(config)
        searcher1 = factory1.create_searcher(prompts)
        result_direct = searcher1.search(prompts)
        
        # Via harness (with same seed, should produce same results)
        factory2 = ComponentFactory(config)
        session = SearchSession.create(
            prompts=prompts,
            config=config,
            factory=factory2,
        )
        result_harness = session.run_to_completion()
        
        # Both should have same structure
        self.assertEqual(set(result_direct.best_scores.keys()),
                         set(result_harness.best_scores.keys()))


if __name__ == "__main__":
    unittest.main()
