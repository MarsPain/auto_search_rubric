"""HTTP client for AutoSR RM server scoring endpoints.

Designed to be imported by external RL repos (e.g. verl) for reward computation.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("reward_harness.rl.verl.reward_client")


try:
    import urllib.request
    import urllib.error
except ImportError:  # pragma: no cover
    urllib = None  # type: ignore[assignment]


try:
    import json
except ImportError:  # pragma: no cover
    json = None  # type: ignore[assignment]


class RMHealthzError(Exception):
    """Raised when RM server health check fails or returns inconsistent metadata."""


class ScoreError(Exception):
    """Raised when a scoring request fails."""


class RMScoringClient:
    """Minimal HTTP client for AutoSR RM server.

    This client intentionally avoids heavy dependencies (no ``requests``)
    so it can be vendored into external RL repos with minimal friction.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        expected_artifact_id: str = "",
        expected_source_session_id: str = "",
        expected_schema_version: str = "1.0",
        expected_rm_api_version: str = "1.0",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.expected_artifact_id = expected_artifact_id
        self.expected_source_session_id = expected_source_session_id
        self.expected_schema_version = expected_schema_version
        self.expected_rm_api_version = expected_rm_api_version
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if urllib is None or json is None:  # pragma: no cover
            raise RuntimeError("standard library urllib/json required")

        url = f"{self.endpoint}{path}"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ScoreError(f"HTTP {exc.code} from {url}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ScoreError(f"Connection error to {url}: {exc.reason}") from exc
        except Exception as exc:
            raise ScoreError(f"Unexpected error calling {url}: {exc}") from exc

    # ------------------------------------------------------------------
    # Healthz / handshake
    # ------------------------------------------------------------------

    def healthz_check(self) -> dict[str, Any]:
        """Call GET /healthz and optionally validate artifact consistency.

        Returns the parsed JSON response. Raises ``RMHealthzError`` on failure
        or when expected artifact/session IDs do not match.
        """
        if urllib is None or json is None:  # pragma: no cover
            raise RuntimeError("standard library urllib/json required")

        url = f"{self.endpoint}/healthz"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, method="GET"),
                timeout=self.timeout_seconds,
            ) as resp:
                data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RMHealthzError(f"HTTP {exc.code} from {url}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RMHealthzError(f"Connection error to {url}: {exc.reason}") from exc
        except Exception as exc:
            raise RMHealthzError(f"Unexpected error calling {url}: {exc}") from exc

        status = str(data.get("status", "")).strip()
        if status != "ok":
            raise RMHealthzError(f"healthz status is not ok: {status!r}")

        # Consistency validation
        if self.expected_artifact_id:
            actual = data.get("artifact_id", "")
            if actual != self.expected_artifact_id:
                raise RMHealthzError(
                    f"artifact_id mismatch: expected {self.expected_artifact_id!r}, "
                    f"got {actual!r}"
                )

        if self.expected_source_session_id:
            actual = data.get("source_session_id", "")
            if actual != self.expected_source_session_id:
                raise RMHealthzError(
                    f"source_session_id mismatch: expected {self.expected_source_session_id!r}, "
                    f"got {actual!r}"
                )

        if self.expected_schema_version:
            actual = str(data.get("schema_version", "")).strip()
            if actual != self.expected_schema_version:
                raise RMHealthzError(
                    f"schema_version mismatch: expected {self.expected_schema_version!r}, "
                    f"got {actual!r}"
                )

        if self.expected_rm_api_version:
            actual = str(data.get("rm_api_version", "")).strip()
            if actual != self.expected_rm_api_version:
                raise RMHealthzError(
                    f"rm_api_version mismatch: expected {self.expected_rm_api_version!r}, "
                    f"got {actual!r}"
                )

        logger.info(
            "RM healthz ok: artifact_id=%s source_session_id=%s schema=%s rm_api=%s",
            data.get("artifact_id"),
            data.get("source_session_id"),
            data.get("schema_version"),
            data.get("rm_api_version"),
        )
        return data

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        *,
        prompt_id: str,
        prompt: str,
        candidate_id: str,
        text: str,
        source: str = "verl",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score a single candidate via POST /score."""
        payload = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "candidate": {
                "candidate_id": candidate_id,
                "text": text,
                "source": source,
                "metadata": metadata or {},
            },
        }
        return self._post_json("/score", payload)

    def batch_score(
        self,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Score a batch of candidates via POST /batch_score.

        Each item in ``items`` must be a dict with keys:
        ``prompt_id``, ``prompt``, ``candidate`` (dict with ``candidate_id``, ``text``).
        """
        payload = {"items": items}
        return self._post_json("/batch_score", payload)
