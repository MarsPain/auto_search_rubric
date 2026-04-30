"""CLI shim: ``python -m reward_harness.rl.verl.run_verl_training``.

Delegates to ``autosr.rl.verl.run_verl_training``.
"""

from __future__ import annotations

from autosr.rl.verl.run_verl_training import main

if __name__ == "__main__":
    main()
