from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from reward_harness.data_models import (
    Criterion,
    GradingProtocol,
    ResponseCandidate,
    Rubric,
)
from reward_harness.evaluator import RubricEvaluator
from reward_harness.rm.data_models import ArtifactValidationError, RMArtifact
from reward_harness.rm.io import save_rm_artifact
from reward_harness.rm.service import (
    RMScoringService,
    RMServerRuntime,
    RequestAuditLogger,
    build_runtime_from_artifact,
    create_app,
)


def _build_test_artifact(*, with_runtime_snapshot: bool = True) -> RMArtifact:
    runtime_snapshot = {}
    if with_runtime_snapshot:
        runtime_snapshot = {
            "seed": 13,
            "extraction": {
                "strategy": "identity",
                "tag_name": "content",
                "pattern": None,
                "join_separator": "\n\n",
            },
            "candidate_extraction": {
                "strategy": "identity",
                "join_separator": "\n\n",
            },
            "llm": {
                "base_url": "https://example.invalid/v1",
                "timeout": 30.0,
                "max_retries": 1,
                "retry_backoff_base": 0.5,
                "retry_backoff_max": 2.0,
                "retry_jitter": 0.1,
                "fail_soft": False,
                "default_model": "model-default",
                "verifier_model": "model-verifier",
                "prompt_language": "zh",
            },
        }

    rubric = Rubric(
        rubric_id="r_server",
        criteria=[
            Criterion(criterion_id="c1", text="quality", weight=0.7),
            Criterion(criterion_id="c2", text="accuracy", weight=0.3),
        ],
        grading_protocol=GradingProtocol(num_votes=3),
    )
    return RMArtifact(
        artifact_id="rm_server_001",
        source_session_id="session_123",
        source_run_id="run_123",
        dataset_hash="dataset_hash_123",
        config_hash="config_hash_123",
        rubric={"p1": rubric},
        scoring_policy={"policy_version": "1.0"},
        normalization={"method": "identity", "score_range": [0.0, 1.0]},
        compatibility={"artifact_type": "rm", "min_rm_api_version": "1.0"},
        runtime_snapshot=runtime_snapshot,
    )


class _DeterministicVerifier:
    def grade(
        self,
        prompt: str,
        candidate: ResponseCandidate,
        rubric: Rubric,
        *,
        seed: int,
    ) -> dict[str, float | None]:
        del prompt, candidate
        return {
            criterion.criterion_id: float((seed + idx) % 6)
            for idx, criterion in enumerate(rubric.criteria)
        }


class TestRMScoringService(unittest.TestCase):
    def test_score_matches_evaluator_ranking_path(self) -> None:
        artifact = _build_test_artifact()
        evaluator = RubricEvaluator(_DeterministicVerifier(), base_seed=13)
        service = RMScoringService(artifact=artifact, evaluator=evaluator)
        rubric = artifact.rubric["p1"]

        candidates = [
            ResponseCandidate(candidate_id="c1", text="alpha"),
            ResponseCandidate(candidate_id="c2", text="beta"),
        ]
        ranked = [
            evaluator.evaluate_single_candidate(
                prompt_id="p1",
                prompt="Prompt",
                candidate=candidate,
                rubric=rubric,
                use_cache=False,
            )
            for candidate in candidates
        ]
        ranked.sort(key=lambda ev: ev.score, reverse=True)

        for expected in ranked:
            response = service.score(
                prompt_id="p1",
                prompt="Prompt",
                candidate=next(
                    c for c in candidates if c.candidate_id == expected.candidate_id
                ),
            )
            self.assertAlmostEqual(response["score"], expected.score, places=8)
            self.assertEqual(response["majority_grades"], expected.majority_grades)
            self.assertEqual(response["vote_scores"], expected.vote_scores)
            self.assertEqual(response["per_vote_grades"], expected.per_vote_grades)


class TestRMServerApi(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = Path(self.temp_dir) / "requests.jsonl"
        artifact = _build_test_artifact()
        evaluator = RubricEvaluator(_DeterministicVerifier(), base_seed=13)
        runtime = RMServerRuntime(
            artifact=artifact,
            scoring_service=RMScoringService(artifact=artifact, evaluator=evaluator),
            request_logger=RequestAuditLogger(self.log_path),
        )
        self.client = TestClient(create_app(runtime))

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_healthz(self) -> None:
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["artifact_id"], "rm_server_001")
        self.assertEqual(payload["source_session_id"], "session_123")
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["rm_api_version"], "1.0")

    def test_score_success(self) -> None:
        response = self.client.post(
            "/score",
            json={
                "prompt_id": "p1",
                "prompt": "Prompt",
                "candidate": {
                    "candidate_id": "c1",
                    "text": "answer",
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("request_id", payload)
        self.assertEqual(payload["artifact_id"], "rm_server_001")
        self.assertEqual(payload["prompt_id"], "p1")
        self.assertEqual(payload["candidate_id"], "c1")
        self.assertIn("score", payload)
        self.assertIn("per_vote_grades", payload)

    def test_score_unknown_prompt_returns_404(self) -> None:
        response = self.client.post(
            "/score",
            json={
                "prompt_id": "missing_prompt",
                "prompt": "Prompt",
                "candidate": {
                    "candidate_id": "c1",
                    "text": "answer",
                },
            },
        )
        self.assertEqual(response.status_code, 404)
        payload = response.json()
        self.assertEqual(payload["detail"]["error_code"], "prompt_not_found")

    def test_batch_score_partial_failure(self) -> None:
        response = self.client.post(
            "/batch_score",
            json={
                "items": [
                    {
                        "prompt_id": "p1",
                        "prompt": "Prompt",
                        "candidate": {
                            "candidate_id": "c1",
                            "text": "answer",
                        },
                    },
                    {
                        "prompt_id": "missing_prompt",
                        "prompt": "Prompt",
                        "candidate": {
                            "candidate_id": "c2",
                            "text": "answer",
                        },
                    },
                ]
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["results"]), 2)
        self.assertTrue(payload["results"][0]["ok"])
        self.assertFalse(payload["results"][1]["ok"])
        self.assertEqual(payload["results"][1]["error_code"], "prompt_not_found")

    def test_request_logs_include_request_id(self) -> None:
        response = self.client.post(
            "/score",
            json={
                "prompt_id": "p1",
                "prompt": "Prompt",
                "candidate": {
                    "candidate_id": "c1",
                    "text": "answer",
                },
            },
        )
        request_id = response.json()["request_id"]
        lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(lines), 1)
        last_event = json.loads(lines[-1])
        self.assertEqual(last_event["request_id"], request_id)
        self.assertEqual(last_event["endpoint"], "/score")


class TestRMServerStartup(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.artifact_path = Path(self.temp_dir) / "artifact.json"

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_runtime_build_requires_runtime_snapshot(self) -> None:
        artifact = _build_test_artifact(with_runtime_snapshot=False)
        save_rm_artifact(self.artifact_path, artifact)
        with self.assertRaises(ArtifactValidationError):
            build_runtime_from_artifact(
                artifact_path=self.artifact_path,
                api_key_env="LLM_API_KEY",
                request_log_path=Path(self.temp_dir) / "requests.jsonl",
            )

    def test_runtime_build_requires_api_key(self) -> None:
        artifact = _build_test_artifact(with_runtime_snapshot=True)
        save_rm_artifact(self.artifact_path, artifact)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                build_runtime_from_artifact(
                    artifact_path=self.artifact_path,
                    api_key_env="LLM_API_KEY",
                    request_log_path=Path(self.temp_dir) / "requests.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
