"""CLI shim: ``python -m reward_harness.rl.check_regression``.

Delegates to ``autosr.rl.cli.check_regression``.
"""

from __future__ import annotations

from autosr.rl.cli.check_regression import main

if __name__ == "__main__":
    main()
