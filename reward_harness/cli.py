"""CLI entry point for Reward Harness.

Delegates to ``autosr.cli.main`` so both ``python -m reward_harness.cli``
and ``python -m autosr.cli`` execute the same code path.
"""

from __future__ import annotations

from autosr.cli import main

if __name__ == "__main__":
    main()
