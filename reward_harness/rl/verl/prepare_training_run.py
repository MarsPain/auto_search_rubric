"""CLI shim: ``python -m reward_harness.rl.verl.prepare_training_run``.

Delegates to ``autosr.rl.verl.prepare_training_run``.
"""

from __future__ import annotations

from autosr.rl.verl.prepare_training_run import main

if __name__ == "__main__":
    main()
