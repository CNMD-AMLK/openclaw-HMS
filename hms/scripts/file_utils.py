"""
HMS v2 — File utilities for safe concurrent access.

Provides:
  - Atomic JSON write (write tmp + os.replace)
  - File-level locking via fcntl (Linux) / fallback no-op (Windows)
  - Safe JSONL append
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, List


@contextmanager
def file_lock(path: str, mode: str = "exclusive"):
    """
    File-level lock using fcntl.flock.
    Yields the lock file descriptor. Releases on exit.
    """
    lock_path = path + ".lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def atomic_write_json(path: str, data: Any) -> None:
    """Write JSON to path atomically via tmp file + os.replace."""
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up tmp file on failure (catches BaseException to handle KeyboardInterrupt too)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def safe_read_json(path: str, default: Any = None) -> Any:
    """Read JSON from path, returning default on any error."""
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError, ValueError):
        return default


def safe_append_jsonl(path: str, entry: Dict[str, Any]) -> None:
    """Append a JSON line to a JSONL file atomically with lock."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with file_lock(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def safe_read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read all entries from a JSONL file with lock."""
    if not os.path.isfile(path):
        return []
    with file_lock(path):
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries


def safe_clear_jsonl(path: str) -> None:
    """Truncate a JSONL file with lock."""
    with file_lock(path):
        open(path, "w").close()
