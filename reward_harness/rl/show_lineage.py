"""CLI shim: ``python -m reward_harness.rl.show_lineage``.

Delegates to ``autosr.rl.cli.show_lineage``.
"""

from __future__ import annotations

from autosr.rl.cli.show_lineage import main

if __name__ == "__main__":
    main()
