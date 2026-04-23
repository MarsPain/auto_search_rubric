from __future__ import annotations

"""Backward-compatible CLI entrypoint for comparing two training runs."""

from .cli.compare_runs import main


if __name__ == "__main__":
    main()
