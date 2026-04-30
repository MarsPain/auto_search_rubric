"""CLI shim: ``python -m reward_harness.rl.compare_artifacts``.

Delegates to ``autosr.rl.cli.compare_artifacts``.
"""

from __future__ import annotations

from autosr.rl.cli.compare_artifacts import main

if __name__ == "__main__":
    main()
