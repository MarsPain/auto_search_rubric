"""CLI shim: ``python -m reward_harness.rl.record_result``.

Delegates to ``autosr.rl.cli.record_result``.
"""

from __future__ import annotations

from autosr.rl.cli.record_result import main

if __name__ == "__main__":
    main()
