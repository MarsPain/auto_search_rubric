"""CLI shim: ``python -m reward_harness.rm.export``.

Delegates to ``autosr.rm.export``.
"""

from __future__ import annotations

from autosr.rm.export import main

if __name__ == "__main__":
    main()
