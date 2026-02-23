from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any

from autosr.config import RuntimeConfig
from autosr.types import LLMRole

_PATH_ARGS = {"--dataset", "--output", "--preset-rubrics"}


def _to_jsonable(value: Any) -> Any:
    """Convert nested values to JSON-safe primitives."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _to_absolute_path(path_token: str, repo_root: Path) -> str:
    path = Path(path_token).expanduser()
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


def _normalize_cli_args(raw_cli_args: list[str], repo_root: Path) -> list[str]:
    normalized: list[str] = []
    idx = 0
    while idx < len(raw_cli_args):
        token = raw_cli_args[idx]

        matched_path_assignment = False
        for path_arg in _PATH_ARGS:
            prefix = f"{path_arg}="
            if token.startswith(prefix):
                path_token = token[len(prefix):]
                normalized.append(f"{path_arg}={_to_absolute_path(path_token, repo_root)}")
                matched_path_assignment = True
                break
        if matched_path_assignment:
            idx += 1
            continue

        if token in _PATH_ARGS and idx + 1 < len(raw_cli_args):
            normalized.append(token)
            normalized.append(_to_absolute_path(raw_cli_args[idx + 1], repo_root))
            idx += 2
            continue

        normalized.append(token)
        idx += 1
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git_command(repo_root: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return completed.stdout.strip()


def _collect_git_snapshot(repo_root: Path) -> dict[str, Any]:
    commit = _run_git_command(repo_root, ["rev-parse", "HEAD"])
    status = _run_git_command(repo_root, ["status", "--porcelain"])
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def build_run_manifest(
    *,
    args: Any,
    config: RuntimeConfig,
    dataset_path: Path,
    output_path: Path,
    raw_cli_args: list[str],
    run_id: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build a reproducibility manifest for the current run."""
    resolved_repo_root = (repo_root or Path.cwd()).resolve()
    resolved_dataset_path = (
        dataset_path if dataset_path.is_absolute() else (resolved_repo_root / dataset_path).resolve()
    )
    resolved_output_path = (
        output_path if output_path.is_absolute() else (resolved_repo_root / output_path).resolve()
    )
    resolved_backend = config.resolve_backend()
    normalized_argv = _normalize_cli_args(raw_cli_args, resolved_repo_root)

    config_snapshot = {
        "search": _to_jsonable(config.search),
        "objective": _to_jsonable(config.objective),
        "initializer": _to_jsonable(config.initializer),
        "extraction": _to_jsonable(config.extraction),
        "verifier": _to_jsonable(config.verifier),
    }
    llm_snapshot = {
        "base_url": config.llm.base_url,
        "timeout": config.llm.timeout,
        "max_retries": config.llm.max_retries,
        "default_model": config.llm.default_model,
        "initializer_model": config.llm.get_model_for_role(LLMRole.INITIALIZER),
        "proposer_model": config.llm.get_model_for_role(LLMRole.PROPOSER),
        "verifier_model": config.llm.get_model_for_role(LLMRole.VERIFIER),
        "judge_model": config.llm.get_model_for_role(LLMRole.JUDGE),
    }

    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": {
            "repo_root": str(resolved_repo_root),
            "cwd": str(Path.cwd().resolve()),
        },
        "dataset": {
            "path": str(resolved_dataset_path),
            "dataset_sha256": _sha256_file(resolved_dataset_path),
        },
        "output": {
            "path": str(resolved_output_path),
        },
        "backend": {
            "requested": str(config.backend),
            "resolved": str(resolved_backend),
            "api_key_env": args.api_key_env,
            "api_key_present": bool(config.llm.api_key),
        },
        "seed": config.search.seed,
        "config_snapshot": config_snapshot,
        "llm_snapshot": llm_snapshot,
        "code_snapshot": _collect_git_snapshot(resolved_repo_root),
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "command": {
            "raw_argv": list(raw_cli_args),
            "normalized_argv": normalized_argv,
        },
    }


def build_reproducible_script(run_manifest: dict[str, Any]) -> str:
    """Build a runnable shell script that replays the experiment command."""
    workspace = run_manifest.get("workspace", {})
    backend = run_manifest.get("backend", {})
    command = run_manifest.get("command", {})
    repo_root = str(workspace.get("repo_root", "."))
    resolved_backend = str(backend.get("resolved", "mock"))
    api_key_env = str(backend.get("api_key_env", "LLM_API_KEY"))
    argv = [str(token) for token in command.get("normalized_argv", [])]

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(repo_root)}",
        "",
    ]
    if resolved_backend == "llm":
        lines.extend([
            f'if [[ -z "${{{api_key_env}:-}}" ]]; then',
            f'  echo "{api_key_env} is not set. Export it before running this script." >&2',
            "  exit 1",
            "fi",
            "",
        ])

    lines.extend([
        "cmd=(",
        "  python3",
        "  -m",
        "  autosr.cli",
    ])
    for token in argv:
        lines.append(f"  {shlex.quote(token)}")
    lines.extend([
        ")",
        'echo "Running command: ${cmd[*]}"',
        '"${cmd[@]}"',
        "",
    ])
    return "\n".join(lines)
