"""CLI shim: ``python -m reward_harness.rm.server``.

Delegates to ``autosr.rm.server``.
"""

from __future__ import annotations

from autosr.rm.server import main

if __name__ == "__main__":
    main()
