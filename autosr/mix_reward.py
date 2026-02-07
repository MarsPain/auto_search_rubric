from __future__ import annotations

from dataclasses import dataclass


def blended_reward(conservative_reward: float, aggressive_reward: float, eta: float) -> float:
    eta_clamped = min(1.0, max(0.0, eta))
    return ((1.0 - eta_clamped) * conservative_reward) + (eta_clamped * aggressive_reward)


@dataclass(slots=True)
class EtaScheduler:
    warmup_steps: int = 0
    ramp_steps: int = 1_000
    start: float = 0.0
    end: float = 1.0

    def value(self, step: int) -> float:
        if step <= self.warmup_steps:
            return self.start
        if self.ramp_steps <= 0:
            return self.end
        progress = (step - self.warmup_steps) / float(self.ramp_steps)
        progress = min(1.0, max(0.0, progress))
        return self.start + ((self.end - self.start) * progress)

