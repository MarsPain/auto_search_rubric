"""AutoSR RL integration reference adapters for external RL frameworks.

This subpackage provides reference implementations for connecting external
RL training repositories (e.g. verl) to the AutoSR RM server and experiment
registry. It is not a mandatory dependency; external projects may reimplement
these patterns in their own codebase.

Reference flow:
1. prepare_training_run  -> generate TrainingManifest + healthz handshake
2. run_training          -> external trainer consumes RM endpoint
3. finalize_training_run -> generate TrainingResultManifest + EvalReport
"""

from .reward_client import RMHealthzError, RMScoringClient, ScoreError

__all__ = [
    "RMScoringClient",
    "RMHealthzError",
    "ScoreError",
]
