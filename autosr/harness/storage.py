"""Persistent storage for search checkpoints.

This module provides atomic file-based checkpoint persistence with
optional time-based and generation-based checkpoint strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from .state import SearchCheckpoint

logger = logging.getLogger("autosr.harness")


@dataclass(slots=True)
class CheckpointMetadata:
    """Metadata about a stored checkpoint."""
    session_id: str
    generation: int
    checkpoint_path: Path
    created_at_utc: str
    file_size_bytes: int


class CheckpointNotFoundError(Exception):
    """Raised when a requested checkpoint cannot be found."""
    pass


class CheckpointCorruptedError(Exception):
    """Raised when checkpoint data is corrupted or unreadable."""
    pass


class StateManager:
    """Manager for checkpoint persistence operations.
    
    Provides atomic file-based storage for search checkpoints with:
    - Atomic writes (write to tmp file, then rename)
    - Automatic checkpoint directory organization
    - Checkpoint listing and retrieval by session_id
    
    Usage:
        state_manager = StateManager(base_dir="./checkpoints")
        
        # Save checkpoint
        state_manager.save_checkpoint(checkpoint)
        
        # Load latest checkpoint for a session
        checkpoint = state_manager.load_checkpoint(session_id="session_001")
        
        # Load specific checkpoint by path
        checkpoint = state_manager.load_checkpoint(path="./checkpoints/session_001/gen_0005.json")
    """
    
    def __init__(
        self,
        base_dir: str | Path = "./checkpoints",
    ) -> None:
        """Initialize state manager.
        
        Args:
            base_dir: Base directory for checkpoint storage.
                     Checkpoints will be stored under `<base_dir>/<session_id>/`
        """
        self.base_dir = Path(base_dir).resolve()
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self) -> None:
        """Ensure the base directory exists."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_session_dir(self, session_id: str) -> Path:
        """Get the checkpoint directory for a session."""
        return self.base_dir / session_id
    
    def _get_checkpoint_path(self, session_id: str, generation: int) -> Path:
        """Get the file path for a checkpoint."""
        session_dir = self._get_session_dir(session_id)
        return session_dir / f"gen_{generation:04d}.json"
    
    def save_checkpoint(
        self,
        checkpoint: SearchCheckpoint,
        *,
        atomic: bool = True,
    ) -> Path:
        """Save checkpoint to persistent storage.
        
        Args:
            checkpoint: The checkpoint to save
            atomic: If True, use atomic write (tmp file + rename)
            
        Returns:
            Path to the saved checkpoint file
        """
        session_dir = self._get_session_dir(checkpoint.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = self._get_checkpoint_path(
            checkpoint.session_id,
            checkpoint.generation,
        )
        
        json_content = checkpoint.to_json(indent=2)
        
        if atomic:
            # Atomic write: write to temp file, then rename
            fd, temp_path = tempfile.mkstemp(
                dir=session_dir,
                prefix=f"gen_{checkpoint.generation:04d}_tmp_",
                suffix=".json",
            )
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(json_content)
                os.rename(temp_path, checkpoint_path)
            except Exception:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
        else:
            checkpoint_path.write_text(json_content, encoding='utf-8')
        
        logger.debug(
            "Checkpoint saved session_id=%s generation=%d path=%s",
            checkpoint.session_id,
            checkpoint.generation,
            checkpoint_path,
        )
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        *,
        session_id: str | None = None,
        path: str | Path | None = None,
    ) -> SearchCheckpoint:
        """Load checkpoint from persistent storage.
        
        Args:
            session_id: Load the latest checkpoint for this session
            path: Load checkpoint from specific file path
            
        Returns:
            Loaded SearchCheckpoint
            
        Raises:
            ValueError: If neither session_id nor path is provided
            CheckpointNotFoundError: If checkpoint file not found
            CheckpointCorruptedError: If checkpoint data is invalid
        """
        if path is not None:
            checkpoint_path = Path(path)
        elif session_id is not None:
            checkpoint_path = self._find_latest_checkpoint(session_id)
            if checkpoint_path is None:
                raise CheckpointNotFoundError(
                    f"No checkpoints found for session: {session_id}"
                )
        else:
            raise ValueError("Either session_id or path must be provided")
        
        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            json_content = checkpoint_path.read_text(encoding='utf-8')
            checkpoint = SearchCheckpoint.from_json(json_content)
        except json.JSONDecodeError as e:
            raise CheckpointCorruptedError(
                f"Failed to parse checkpoint JSON: {checkpoint_path}"
            ) from e
        except Exception as e:
            raise CheckpointCorruptedError(
                f"Failed to load checkpoint: {checkpoint_path}"
            ) from e
        
        logger.debug(
            "Checkpoint loaded session_id=%s generation=%d path=%s",
            checkpoint.session_id,
            checkpoint.generation,
            checkpoint_path,
        )
        
        return checkpoint
    
    def _find_latest_checkpoint(self, session_id: str) -> Path | None:
        """Find the latest checkpoint for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Path to latest checkpoint, or None if no checkpoints found
        """
        session_dir = self._get_session_dir(session_id)
        if not session_dir.exists():
            return None
        
        checkpoint_files = sorted(session_dir.glob("gen_*.json"))
        if not checkpoint_files:
            return None
        
        # Return the last one (sorted by generation number)
        return checkpoint_files[-1]
    
    def list_checkpoints(self, session_id: str) -> list[CheckpointMetadata]:
        """List all checkpoints for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of checkpoint metadata, sorted by generation
        """
        session_dir = self._get_session_dir(session_id)
        if not session_dir.exists():
            return []
        
        checkpoints: list[CheckpointMetadata] = []
        
        for checkpoint_path in sorted(session_dir.glob("gen_*.json")):
            try:
                # Parse generation from filename
                generation = int(checkpoint_path.stem.split("_")[1])
                stat = checkpoint_path.stat()
                
                # Try to read metadata from file without full parsing
                checkpoint = self.load_checkpoint(path=checkpoint_path)
                
                checkpoints.append(CheckpointMetadata(
                    session_id=session_id,
                    generation=generation,
                    checkpoint_path=checkpoint_path,
                    created_at_utc=checkpoint.created_at_utc,
                    file_size_bytes=stat.st_size,
                ))
            except Exception as e:
                logger.warning(
                    "Failed to read checkpoint metadata: %s (%s)",
                    checkpoint_path,
                    e,
                )
                continue
        
        return checkpoints
    
    def list_sessions(self) -> list[str]:
        """List all session IDs with checkpoints.
        
        Returns:
            List of session IDs
        """
        if not self.base_dir.exists():
            return []
        
        sessions = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and any(item.glob("gen_*.json")):
                sessions.append(item.name)
        
        return sorted(sessions)
    
    def delete_checkpoint(self, session_id: str, generation: int) -> bool:
        """Delete a specific checkpoint.
        
        Args:
            session_id: Session identifier
            generation: Generation number
            
        Returns:
            True if checkpoint was deleted, False if not found
        """
        checkpoint_path = self._get_checkpoint_path(session_id, generation)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.debug(
                "Checkpoint deleted session_id=%s generation=%d",
                session_id,
                generation,
            )
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete all checkpoints for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session directory was deleted, False if not found
        """
        session_dir = self._get_session_dir(session_id)
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
            logger.debug("Session checkpoints deleted session_id=%s", session_id)
            return True
        return False
