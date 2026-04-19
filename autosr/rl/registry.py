from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any

from .data_models import EvalReport, LineageIndex, TrainingManifest, TrainingResultManifest
from .io import (
    load_eval_report,
    load_lineage_index,
    load_training_manifest,
    load_training_result,
    save_eval_report,
    save_lineage_index,
    save_training_manifest,
    save_training_result,
)

logger = logging.getLogger("autosr.rl.registry")
_SAFE_ENTRY_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class RegistryError(Exception):
    """Raised when registry operations fail."""


class DuplicateEntryError(RegistryError):
    """Raised when attempting to overwrite an existing entry without force."""


class MissingManifestError(RegistryError):
    """Raised when a required TrainingManifest does not exist."""


class ExperimentRegistry:
    """Append-only experiment registry for RL training metadata.

    Directory layout under base_dir:
        manifests/   -> TrainingManifest files
        results/     -> TrainingResultManifest files
        evals/       -> EvalReport files
        index/       -> LineageIndex files (derived, optional)
    """

    def __init__(self, base_dir: str | Path = "artifacts/training_runs") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.manifests_dir = self.base_dir / "manifests"
        self.results_dir = self.base_dir / "results"
        self.evals_dir = self.base_dir / "evals"
        self.index_dir = self.base_dir / "index"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.evals_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self, training_run_id: str) -> Path:
        return self._entry_path(self.manifests_dir, training_run_id, field_name="training_run_id")

    def _result_path(self, training_run_id: str) -> Path:
        return self._entry_path(self.results_dir, training_run_id, field_name="training_run_id")

    def _eval_path(self, eval_run_id: str) -> Path:
        return self._entry_path(self.evals_dir, eval_run_id, field_name="eval_run_id")

    def _index_path(self, training_run_id: str) -> Path:
        return self._entry_path(self.index_dir, training_run_id, field_name="training_run_id")

    def _entry_path(self, root_dir: Path, entry_id: str, *, field_name: str) -> Path:
        normalized = entry_id.strip()
        if not normalized:
            raise RegistryError(f"{field_name} must not be empty")
        if not _SAFE_ENTRY_ID_RE.fullmatch(normalized):
            raise RegistryError(
                f"{field_name} contains unsafe characters: {entry_id!r}; "
                "only [A-Za-z0-9._-] are allowed"
            )
        resolved_root = root_dir.resolve()
        resolved_path = (resolved_root / f"{normalized}.json").resolve()
        if not resolved_path.is_relative_to(resolved_root):
            raise RegistryError(
                f"{field_name} resolves outside registry directory: {entry_id!r}"
            )
        return resolved_path

    # ------------------------------------------------------------------
    # Manifests
    # ------------------------------------------------------------------

    def record_manifest(
        self,
        manifest: TrainingManifest,
        *,
        allow_identical_overwrite: bool = True,
    ) -> Path:
        """Record a TrainingManifest.

        By default (allow_identical_overwrite=True), identical payloads are treated
        as idempotent. Set allow_identical_overwrite=False to reject any existing entry.
        """
        path = self._manifest_path(manifest.training_run_id)
        if path.exists():
            if not allow_identical_overwrite:
                raise DuplicateEntryError(
                    f"TrainingManifest already exists for {manifest.training_run_id}"
                )
            existing = load_training_manifest(path)
            if existing.to_dict() == manifest.to_dict():
                logger.info(
                    "Manifest identical for training_run_id=%s; treating as idempotent",
                    manifest.training_run_id,
                )
                return path
            raise DuplicateEntryError(
                f"TrainingManifest already exists for {manifest.training_run_id} "
                f"and payload differs"
            )
        save_training_manifest(path, manifest)
        self._sync_index(manifest.training_run_id)
        logger.info("Recorded TrainingManifest training_run_id=%s", manifest.training_run_id)
        return path

    def get_manifest(self, training_run_id: str) -> TrainingManifest | None:
        path = self._manifest_path(training_run_id)
        if not path.exists():
            return None
        return load_training_manifest(path)

    def list_manifests(self) -> list[str]:
        return sorted(p.stem for p in self.manifests_dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def record_result(
        self,
        result: TrainingResultManifest,
        *,
        allow_overwrite: bool = False,
    ) -> Path:
        """Record a TrainingResultManifest. Requires a prior manifest."""
        manifest = self.get_manifest(result.training_run_id)
        if manifest is None:
            raise MissingManifestError(
                f"Cannot record result for {result.training_run_id}: "
                "TrainingManifest not found"
            )
        path = self._result_path(result.training_run_id)
        if path.exists():
            if allow_overwrite:
                raise DuplicateEntryError(
                    f"TrainingResultManifest already exists for {result.training_run_id}; "
                    "append-only registry does not allow overwrite"
                )
            raise DuplicateEntryError(
                f"TrainingResultManifest already exists for {result.training_run_id}"
            )
        save_training_result(path, result)
        self._sync_index(result.training_run_id)
        logger.info("Recorded TrainingResultManifest training_run_id=%s", result.training_run_id)
        return path

    def get_result(self, training_run_id: str) -> TrainingResultManifest | None:
        path = self._result_path(training_run_id)
        if not path.exists():
            return None
        return load_training_result(path)

    def list_results(self) -> list[str]:
        return sorted(p.stem for p in self.results_dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Evals
    # ------------------------------------------------------------------

    def record_eval(self, report: EvalReport) -> Path:
        """Record an EvalReport. Requires a prior manifest."""
        manifest = self.get_manifest(report.training_run_id)
        if manifest is None:
            raise MissingManifestError(
                f"Cannot record eval for {report.training_run_id}: "
                "TrainingManifest not found"
            )
        path = self._eval_path(report.eval_run_id)
        if path.exists():
            existing = load_eval_report(path)
            if existing.to_dict() == report.to_dict():
                logger.info(
                    "EvalReport identical for eval_run_id=%s; treating as idempotent",
                    report.eval_run_id,
                )
                return path
            raise DuplicateEntryError(
                f"EvalReport already exists for {report.eval_run_id} and payload differs"
            )
        save_eval_report(path, report)
        self._sync_index(report.training_run_id)
        logger.info("Recorded EvalReport eval_run_id=%s", report.eval_run_id)
        return path

    def get_eval(self, eval_run_id: str) -> EvalReport | None:
        path = self._eval_path(eval_run_id)
        if not path.exists():
            return None
        return load_eval_report(path)

    def list_evals(self) -> list[str]:
        return sorted(p.stem for p in self.evals_dir.glob("*.json"))

    def list_evals_for_training_run(self, training_run_id: str) -> list[str]:
        index = self._load_index(training_run_id)
        if index is not None:
            return list(index.eval_run_ids)
        # Fallback: scan evals dir
        return sorted(
            p.stem
            for p in self.evals_dir.glob("*.json")
            if self.get_eval(p.stem) is not None
            and self.get_eval(p.stem).training_run_id == training_run_id
        )

    # ------------------------------------------------------------------
    # Index / Lineage
    # ------------------------------------------------------------------

    def _load_index(self, training_run_id: str) -> LineageIndex | None:
        path = self._index_path(training_run_id)
        if not path.exists():
            return None
        return load_lineage_index(path)

    def _sync_index(self, training_run_id: str) -> Path:
        manifest = self.get_manifest(training_run_id)
        if manifest is None:
            # If manifest was deleted externally, keep existing index or skip
            return self._index_path(training_run_id)
        eval_run_ids = []
        for p in self.evals_dir.glob("*.json"):
            try:
                report = load_eval_report(p)
                if report.training_run_id == training_run_id:
                    eval_run_ids.append(report.eval_run_id)
            except Exception:
                continue
        index = LineageIndex(
            training_run_id=training_run_id,
            rm_artifact_id=manifest.rm_artifact_id,
            rm_deploy_id=manifest.rm_deploy_id,
            search_session_id=manifest.search_session_id,
            eval_run_ids=sorted(set(eval_run_ids)),
        )
        path = self._index_path(training_run_id)
        save_lineage_index(path, index)
        return path

    def get_lineage_index(self, training_run_id: str) -> LineageIndex | None:
        return self._load_index(training_run_id)

    def list_training_run_ids(self) -> list[str]:
        return self.list_manifests()

    # ------------------------------------------------------------------
    # Full lineage resolution
    # ------------------------------------------------------------------

    def resolve_lineage(self, training_run_id: str) -> dict[str, Any] | None:
        """Resolve full lineage for a training run.

        Returns a dict with keys:
            training_run_id, manifest, result, evals, upstream
        or None if manifest missing.
        """
        manifest = self.get_manifest(training_run_id)
        if manifest is None:
            return None
        result = self.get_result(training_run_id)
        eval_ids = self.list_evals_for_training_run(training_run_id)
        evals = [self.get_eval(eid) for eid in eval_ids]
        return {
            "training_run_id": training_run_id,
            "manifest": manifest,
            "result": result,
            "evals": [ev for ev in evals if ev is not None],
            "upstream": {
                "rm_artifact_id": manifest.rm_artifact_id,
                "rm_deploy_id": manifest.rm_deploy_id,
                "search_session_id": manifest.search_session_id,
            },
        }
