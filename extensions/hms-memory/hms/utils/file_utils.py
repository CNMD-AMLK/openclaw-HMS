"""
HMS v5 — File utilities (adapted from hms-v4 scripts/file_utils.py).
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
_lock_fd_counts: Dict[int, int] = {}
_lock_fds_lock = threading.Lock()
_thread_locks: Dict[str, threading.Lock] = {}
_thread_locks_lock = threading.Lock()


def _get_lock_fd(path: str) -> int:
    if path in _lock_fds:
        return _lock_fds[path]
    lock_path = path + ".lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with _lock_fds_lock:
        if path in _lock_fds:
            return _lock_fds[path]
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        _lock_fds[path] = fd
        _lock_fd_counts[fd] = 1
        return fd


@contextmanager
def file_lock(path: str, mode: str = "exclusive"):
    if _HAS_FCNTL:
        fd = _get_lock_fd(path)
        with _lock_fds_lock:
            _lock_fd_counts[fd] = _lock_fd_counts.get(fd, 0) + 1
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            with _lock_fds_lock:
                _lock_fd_counts[fd] = _lock_fd_counts.get(fd, 0) - 1
                if _lock_fd_counts[fd] <= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    _lock_fd_counts.pop(fd, None)
                    paths_to_remove = [p for p, f in _lock_fds.items() if f == fd]
                    for p in paths_to_remove:
                        _lock_fds.pop(p, None)
            with _lock_fds_lock:
                _lock_fds.pop(path, None)
    else:
        if path not in _thread_locks:
            with _thread_locks_lock:
                if path not in _thread_locks:
                    _thread_locks[path] = threading.Lock()
        lock = _thread_locks[path]
        with lock:
            yield None


def atomic_write_json(path: str, data: Any) -> None:
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
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError, ValueError):
        return default


def safe_append_jsonl(path: str, entry: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with file_lock(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def safe_read_jsonl(path: str) -> List[Dict[str, Any]]:
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
    with file_lock(path):
        with open(path, "w", encoding="utf-8") as f:
            f.truncate(0)
