"""Backward-compatible nested CLI entrypoint for record_manifest."""

from __future__ import annotations

from reward_harness.rl.record_manifest import main

if __name__ == "__main__":
    main()
