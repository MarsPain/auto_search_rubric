"""Backward-compatible nested CLI entrypoint for record_result."""

from __future__ import annotations

from reward_harness.rl.record_result import main

if __name__ == "__main__":
    main()
