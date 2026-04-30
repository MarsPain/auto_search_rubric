"""CLI shim: ``python -m reward_harness.rl.verl.finalize_training_run``.

Delegates to ``autosr.rl.verl.finalize_training_run``.
"""

from __future__ import annotations

from autosr.rl.verl.finalize_training_run import main

if __name__ == "__main__":
    main()
