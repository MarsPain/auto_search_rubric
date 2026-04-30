"""CLI shim: ``python -m reward_harness.rm.deploy``.

Delegates to ``autosr.rm.deploy``.
"""

from __future__ import annotations

from autosr.rm.deploy import main

if __name__ == "__main__":
    main()
