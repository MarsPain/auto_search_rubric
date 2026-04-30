"""CLI shim: ``python -m reward_harness.rl.list_runs``.

Delegates to ``autosr.rl.cli.list_runs``.
"""

from __future__ import annotations

from autosr.rl.cli.list_runs import main

if __name__ == "__main__":
    main()
