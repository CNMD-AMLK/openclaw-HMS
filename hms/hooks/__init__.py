"""
HMS v2 — OpenClaw Hook Integration

Provides hook handlers for OpenClaw integration.
These hooks are called by OpenClaw at specific lifecycle points.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, Optional

from ..scripts.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


# Global instance (initialized on first use)
_manager: Optional[MemoryManager] = None


def get_manager() -> MemoryManager:
    """Get or create the global MemoryManager instance."""
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager


def on_message_received(user_message: str, session_id: str = "") -> Dict[str, Any]:
    """
    Hook: Called when a user message is received.
    
    Usage in OpenClaw:
        openclaw hook register message:received "python3 -m hms.hooks.on_message_received"
    """
    mgr = get_manager()
    return mgr.on_message_received(user_message, session_id)


def on_message_sent(user_message: str, assistant_reply: str, session_id: str = "") -> None:
    """
    Hook: Called after assistant sends a reply.
    
    Usage in OpenClaw:
        openclaw hook register message:sent "python3 -m hms.hooks.on_message_sent"
    """
    mgr = get_manager()
    mgr.on_message_sent(user_message, assistant_reply, session_id)


def before_compaction() -> Dict[str, Any]:
    """
    Hook: Called before context compaction.
    Processes pending entries to free up queue.
    
    Usage in OpenClaw:
        openclaw hook register before:compaction "python3 -m hms.hooks.before_compaction"
    """
    mgr = get_manager()
    return mgr.process_pending()


# CLI interface for testing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m hms.hooks <hook_name> [args...]")
        print("Hooks: on_message_received, on_message_sent, before_compaction")
        sys.exit(1)
    
    hook_name = sys.argv[1]
    
    if hook_name == "on_message_received":
        msg = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read().strip()
        result = on_message_received(msg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif hook_name == "on_message_sent":
        if len(sys.argv) < 4:
            print("Usage: on_message_sent <user_message> <assistant_reply>")
            sys.exit(1)
        on_message_sent(sys.argv[2], sys.argv[3])
        print("OK")
    
    elif hook_name == "before_compaction":
        result = before_compaction()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    else:
        print(f"Unknown hook: {hook_name}")
        sys.exit(1)
