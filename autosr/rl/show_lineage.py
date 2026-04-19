from __future__ import annotations

"""Backward-compatible CLI entrypoint for lineage query."""

from .cli.show_lineage import main


if __name__ == "__main__":
    main()
