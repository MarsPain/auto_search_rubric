"""CLI shim: ``python -m reward_harness.rl.record_eval``.

Delegates to ``autosr.rl.cli.record_eval``.
"""

from __future__ import annotations

from autosr.rl.cli.record_eval import main

if __name__ == "__main__":
    main()
