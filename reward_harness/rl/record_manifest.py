"""CLI shim: ``python -m reward_harness.rl.record_manifest``.

Delegates to ``autosr.rl.cli.record_manifest``.
"""

from __future__ import annotations

from autosr.rl.cli.record_manifest import main

if __name__ == "__main__":
    main()
