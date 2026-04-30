"""Backward-compatible nested CLI entrypoint for record_eval."""

from __future__ import annotations

from reward_harness.rl.record_eval import main

if __name__ == "__main__":
    main()
