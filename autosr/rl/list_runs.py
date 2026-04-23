from __future__ import annotations

"""Backward-compatible CLI entrypoint for listing training runs."""

from .cli.list_runs import main


if __name__ == "__main__":
    main()
