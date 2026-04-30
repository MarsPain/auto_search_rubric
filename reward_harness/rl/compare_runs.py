"""CLI shim: ``python -m reward_harness.rl.compare_runs``.

Delegates to ``autosr.rl.cli.compare_runs``.
"""

from __future__ import annotations

from autosr.rl.cli.compare_runs import main

if __name__ == "__main__":
    main()
