from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RoleModelConfig:
    default: str
    initializer: str | None = None
    proposer: str | None = None
    verifier: str | None = None
    judge: str | None = None

    def __post_init__(self) -> None:
        if not self.default.strip():
            raise ValueError("default model must not be empty")

    def for_role(self, role: str) -> str:
        role_to_model = {
            "initializer": self.initializer,
            "proposer": self.proposer,
            "verifier": self.verifier,
            "judge": self.judge,
        }
        model = role_to_model.get(role)
        if model:
            return model
        return self.default


@dataclass(slots=True)
class LLMConfig:
    base_url: str
    api_key: str
    timeout: float = 30.0
    max_retries: int = 2
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
