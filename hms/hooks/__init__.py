"""
HMS v3 — OpenClaw Integration via Cron/Skill Interface

OpenClaw does not expose a native hook registration API.
Instead, HMS integrates through:

1. Cron jobs (recommended for production):
   - openclaw cron add --schedule "* * * * *" --command "python -m hms process_pending"
   - openclaw cron add --schedule "0 3 * * *" --command "python -m hms consolidate"
   - openclaw cron add --schedule "0 4 * * 0" --command "python -m hms forget"

2. Skill/Plugin interface:
   - Import MemoryManager directly in your OpenClaw skill/plugin
   - Call mgr.on_message_received() / mgr.on_message_sent() from skill handlers

3. Direct CLI usage:
   - python -m hms received "用户消息"
   - python -m hms process_pending
   - python -m hms consolidate
   - python -m hms forget
"""

from __future__ import annotations

import atexit
import json
import logging
import sys
from typing import Any, Dict, Optional

from ..scripts.memory_manager import MemoryManager

__all__ = [
    "get_manager",
    "reset_manager",
    "on_message_received",
    "on_message_sent",
    "process_pending",
    "consolidate",
    "forget",
]

logger = logging.getLogger(__name__)

_manager: Optional[MemoryManager] = None


def _cleanup() -> None:
    """Clean up resources on process exit."""
    global _manager
    if _manager is not None:
        try:
            _manager.close()
        except Exception:
            pass
        _manager = None


atexit.register(_cleanup)


def get_manager() -> MemoryManager:
    """Get or create the global MemoryManager instance."""
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager


def reset_manager() -> None:
    """Reset the global MemoryManager instance, closing resources first."""
    global _manager
    if _manager is not None:
        _manager.close()
    _manager = None


def on_message_received(user_message: str, session_id: str = "") -> Dict[str, Any]:
    """Process an incoming user message.

    Call this from your OpenClaw skill/plugin when a message arrives.
    Returns perception + context data to inject into the agent's reply.
    """
    mgr = get_manager()
    return mgr.on_message_received(user_message, session_id)


def on_message_sent(user_message: str, assistant_reply: str, session_id: str = "") -> None:
    """Queue a completed conversation turn for async processing.

    Call this from your OpenClaw skill/plugin after the assistant replies.
    """
    mgr = get_manager()
    mgr.on_message_sent(user_message, assistant_reply, session_id)


def process_pending() -> Dict[str, Any]:
    """Process all queued conversation turns.

    Should be called periodically via cron (e.g. every minute).
    """
    mgr = get_manager()
    return mgr.process_pending()


def consolidate() -> Dict[str, Any]:
    """Run daily consolidation (compression, fingerprint update, etc.).

    Should be called daily via cron (e.g. 3 AM).
    """
    mgr = get_manager()
    return mgr.consolidate()


def forget() -> Dict[str, Any]:
    """Run weekly forgetting pass.

    Should be called weekly via cron (e.g. Sunday 4 AM).
    """
    mgr = get_manager()
    return mgr.forget()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m hms.hooks <command> [args...]")
        print("Commands: received, sent, process_pending, consolidate, forget")
        print("")
        print("Integration options:")
        print("  1. Cron: openclaw cron add --schedule '* * * * *' --command 'python -m hms process_pending'")
        print("  2. Skill: import hms.hooks; hms.hooks.on_message_received(msg)")
        print("  3. CLI:   python -m hms received '消息'")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "received":
        msg = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read().strip()
        result = on_message_received(msg)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif cmd == "sent":
        if len(sys.argv) < 4:
            print("Usage: sent <user_message> <assistant_reply>")
            sys.exit(1)
        on_message_sent(sys.argv[2], sys.argv[3])
        print("OK")

    elif cmd == "process_pending":
        result = process_pending()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif cmd == "consolidate":
        result = consolidate()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif cmd == "forget":
        result = forget()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
