"""CLI entry point for HMS.

Usage:
    python -m hms received "用户消息"
    python -m hms process_pending
    python -m hms consolidate
    python -m hms forget
    python -m hms health
"""

from __future__ import annotations

from .scripts.memory_manager import main

if __name__ == "__main__":
    main()
