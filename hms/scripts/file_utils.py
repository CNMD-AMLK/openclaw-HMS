"""
HMS v3 — File utilities for safe concurrent access.

Provides:
  - Atomic JSON write (write tmp + os.replace)
  - File-level locking via fcntl (Linux) / fallback no-op (Windows)
  - Safe JSONL append
  - Cached lock file descriptors for reduced I/O overhead
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from typing import Any, Dict, List

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

_lock_fds: Dict[str, int] = {}
_lock_fds_lock = threading.Lock()
_thread_locks: Dict[str, threading.Lock] = {}
_thread_locks_lock = threading.Lock()


def _get_lock_fd(path: str) -> int:
    """Get or create a cached lock file descriptor for the given path."""
    if path not in _lock_fds:
        lock_path = path + ".lock"
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        with _lock_fds_lock:
            if path not in _lock_fds:
                _lock_fds[path] = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    return _lock_fds[path]


@contextmanager
def file_lock(path: str, mode: str = "exclusive"):
    """
    File-level lock using fcntl.flock with cached lock FD.
    Falls back to threading.Lock on Windows (no fcntl support).
    Yields the lock file descriptor. Releases on exit.
    """
    if _HAS_FCNTL:
        fd = _get_lock_fd(path)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    else:
        # Windows fallback: thread-level lock (not cross-process)
        if path not in _thread_locks:
            with _thread_locks_lock:
                if path not in _thread_locks:
                    _thread_locks[path] = threading.Lock()
        lock = _thread_locks[path]
        with lock:
            yield None


def atomic_write_json(path: str, data: Any) -> None:
    """Write JSON to path atomically via tmp file + os.replace."""
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
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
        with open(path, "w", encoding="utf-8") as f:
            f.truncate(0)


def close_all_lock_fds() -> None:
    """Close all cached lock file descriptors. Call on shutdown for clean exit."""
    with _lock_fds_lock:
        for fd in _lock_fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        _lock_fds.clear()
