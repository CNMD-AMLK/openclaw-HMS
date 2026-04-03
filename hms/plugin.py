"""
HMS v4.0.0 — OpenClaw Native Plugin.

Implements the HMS OpenClaw plugin interface so HMS can be loaded
directly by OpenClaw as a first-class plugin (not just a CLI utility).

Hooks registered:
  - on_message: intercept every user/assistant message turn
  - on_heartbeat: periodic maintenance (pending processing, health checks)

Tools registered with OpenClaw:
  - hms_perceive     : instant message perception + context retrieval
  - hms_collide      : run collision detection on pending memories
  - hms_recall       : reconstructive memory recall
  - hms_consolidate  : trigger consolidation cycle
  - hms_context_inject: compose memory-augmented context for agent turn
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Resolve package root so HMS modules are importable even when the plugin
# is loaded via OpenClaw's plugin manager (which may import from a different
# working directory).
# ---------------------------------------------------------------------------
_plugin_dir = Path(__file__).resolve().parent
_root_dir = _plugin_dir.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

from hms.scripts.memory_manager import MemoryManager
from hms.scripts.config_loader import Config
from hms.scripts.reconstructive_recall import ReconstructiveRecaller
from hms.scripts.dream_engine import DreamEngine
from hms.scripts.creative_assoc import CreativeAssociator
from hms.scripts.forgetting import ForgettingEngine, MemoryOverwriter

logger = logging.getLogger("hms.plugin")


# ======================================================================
# Plugin class — OpenClaw native entry point
# ======================================================================

class HMSPlugin:
    """
    OpenClaw native plugin for HMS (Hybrid Memory System).

    Lifecycle:
      register(ctx)  → called once when the plugin is loaded
      on_message(msg, ctx) → called for every message
      on_heartbeat(ctx)    → called periodically by OpenClaw
    """

    def __init__(self) -> None:
        self._mgr: Optional[MemoryManager] = None
        self._recaller: Optional[ReconstructiveRecaller] = None
        self._dream_engine: Optional[DreamEngine] = None
        self._creative_assoc: Optional[CreativeAssociator] = None
        self._overwriter: Optional[MemoryOverwriter] = None
        self._lock = threading.Lock()
        self._ctx: Dict[str, Any] = {}
        self._pending_sent_queue: List[Dict[str, str]] = []  # locally queued sent messages

    # ------------------------------------------------------------------
    # Plugin lifecycle
    # ------------------------------------------------------------------

    def register(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called by OpenClaw when the plugin is loaded.

        Registers HMS tools and stores the plugin context.
        Returns a dict describing the registered tools.
        """
        self._ctx = ctx

        plugin_info = {
            "name": "hms-memory",
            "version": "4.0.0",
            "status": "registered",
            "tools": self._list_tools(),
        }

        logger.info("HMS plugin v4.0.0 registered successfully")
        return plugin_info

    def on_message(self, msg: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenClaw on_message hook.

        For user messages: route to hms_perceive internally.
        For assistant replies: queue for async pending processing.
        """
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            return self._handle_user_message(content, ctx)
        elif role == "assistant":
            return self._handle_assistant_message(content, ctx)

        return {"status": "ignored", "reason": f"unknown role: {role}"}

    def on_heartbeat(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenClaw heartbeat hook.

        Performs lightweight maintenance:
          - Process pending queue items
          - Flush forgetting engine dirty state
          - Run dream engine cycle (consolidate_interval triggered)
        """
        result: Dict[str, Any] = {
            "component": "hms-heartbeat",
            "status": "ok",
        }

        try:
            mgr = self._get_manager()
            pending_report = mgr.process_pending()
            result["pending_processed"] = pending_report.get("processed", 0)
        except Exception as exc:
            logger.debug("Heartbeat pending processing error: %s", exc)
            result["pending_error"] = str(exc)

        try:
            # Flush any dirty forgetting engine state
            if self._mgr is not None and self._mgr.forgetting is not None:
                self._mgr.forgetting.flush()
                result["forgetting_flushed"] = True
        except Exception as exc:
            logger.debug("Heartbeat forgetting flush error: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_manager(self) -> MemoryManager:
        """Lazy-initialize the MemoryManager (thread-safe)."""
        if self._mgr is None:
            with self._lock:
                if self._mgr is None:
                    cfg = Config.get()
                    cache_dir = cfg.get("cache_dir", "cache")
                    # Ensure cache directories exist
                    os.makedirs(cache_dir, exist_ok=True)
                    os.makedirs(os.path.join(cache_dir, "insights"), exist_ok=True)

                    self._mgr = MemoryManager(cfg)
                    self._recaller = ReconstructiveRecaller(cfg)
                    self._dream_engine = DreamEngine(cfg)
                    self._creative_assoc = CreativeAssociator(cfg)
                    self._overwriter = MemoryOverwriter(cfg)

                    logger.info("HMS MemoryManager initialized (cache_dir=%s)", cache_dir)
        return self._mgr

    def _get_recaller(self) -> ReconstructiveRecaller:
        self._get_manager()  # ensure init
        assert self._recaller is not None
        return self._recaller

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def _handle_user_message(
        self, content: str, ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle incoming user message: perceive + recall relevant memories
        and return context to inject.
        """
        try:
            mgr = self._get_manager()
            result = mgr.on_message_received(content)

            # Enrich with reconstructive recall for deeper context
            try:
                recaller = self._get_recaller()
                reconstructed = recaller.recall(content, result.get("perception", {}))
                result["reconstructed_context"] = reconstructed
            except Exception as exc:
                logger.debug("Reconstructive recall error: %s", exc)

            return {
                "status": "processed",
                "perception": result.get("perception", {}),
                "context": result.get("context", {}),
                "reconstructed_context": result.get("reconstructed_context"),
            }
        except Exception as exc:
            logger.error("User message handler error: %s", exc)
            return {"status": "error", "error": str(exc)}

    def _handle_assistant_message(
        self, content: str, ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle assistant reply: write to pending queue for async processing.
        Since the assistant reply doesn't carry the user message directly,
        we store it with a placeholder — the actual pairing happens when
        the pending queue is processed (the user message was already captured).
        """
        try:
            mgr = self._get_manager()
            # We need the user_message — if we don't have it from context,
            # queue it for later pairing
            user_msg = ctx.get("last_user_message", "")
            mgr.on_message_sent(user_msg, content)
            return {"status": "queued", "pending_count": mgr.context.get_pending_count()}
        except Exception as exc:
            logger.error("Assistant message handler error: %s", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # HMS tools (these can be called directly from OpenClaw agent)
    # ------------------------------------------------------------------

    def _list_tools(self) -> List[Dict[str, str]]:
        """Return the list of tools this plugin provides."""
        return [
            {"name": "hms_perceive", "description": "Instant perception + memory context for a message"},
            {"name": "hms_collide", "description": "Run collision detection on pending memories"},
            {"name": "hms_recall", "description": "Reconstructive memory recall with context synthesis"},
            {"name": "hms_consolidate", "description": "Trigger daily consolidation cycle"},
            {"name": "hms_context_inject", "description": "Compose full memory-augmented context"},
        ]

    def hms_perceive(self, message: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Tool: Instant perception + memory context for a message.

        Args:
            message: the user message to perceive
            **kwargs: optional session_id, tier

        Returns:
            Perception result with memories and context.
        """
        mgr = self._get_manager()
        tier = kwargs.get("tier", "")
        if tier:
            cfg = Config.get()
            cfg = MemoryManager._apply_tier(cfg, tier)
            mgr = MemoryManager(cfg)

        return mgr.on_message_received(message)

    def hms_collide(self, text: str, memories: Optional[List[Dict]] = None,
                    **kwargs: Any) -> Dict[str, Any]:
        """
        Tool: Run collision detection on a given text against memories.

        Args:
            text: text to check for collisions
            memories: optional list of existing memories to compare against
            **kwargs: optional threshold

        Returns:
            Collision detection result.
        """
        mgr = self._get_manager()
        if memories is None:
            try:
                memories = mgr.adapter.recall(query=text, top_k=10)
            except Exception:
                memories = []

        # Build a minimal perception dict
        perception = mgr.perception.analyze(text, "", force_heuristic=True)
        collision_result = mgr.collision_engine.collide(perception, memories or [])
        exec_result = mgr.collision_engine.execute_results(
            collision_result,
            memory_store_func=lambda **kw: mgr.adapter.store(**kw),
            gm_record_func=lambda **kw: mgr.adapter.graph_record(**kw),
        )
        return exec_result

    def hms_recall(self, query: str, top_k: int = 5, use_reconstructive: bool = True,
                   **kwargs: Any) -> Dict[str, Any]:
        """
        Tool: Memory recall. Supports both exact and reconstructive recall.

        Args:
            query: search query
            top_k: number of results
            use_reconstructive: if True, use reconstructive recall with LLM synthesis

        Returns:
            Recall results with memories and confidence.
        """
        mgr = self._get_manager()

        if use_reconstructive:
            recaller = self._get_recaller()
            return recaller.recall(query, kwargs.get("context", {}))

        # Direct recall
        try:
            memories = mgr.adapter.recall(query=query, top_k=top_k)
        except Exception as exc:
            return {"error": str(exc), "memories": []}

        return {
            "method": "exact",
            "memories": memories,
            "count": len(memories),
            "query": query,
        }

    def hms_consolidate(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Tool: Trigger a consolidation cycle (compression, fingerprint update,
        dream engine, and creative association).

        Returns:
            Consolidation report.
        """
        mgr = self._get_manager()
        report = mgr.consolidate()

        # Run dream engine for distant associations
        try:
            dream_report = self._dream_engine.run_dream_cycle()
            report["dream_insights"] = dream_report.get("insights_count", 0)
        except Exception as exc:
            logger.debug("Dream engine error during consolidate: %s", exc)
            report["dream_error"] = str(exc)

        # Run creative association
        try:
            creative_report = self._creative_assoc.generate_insights()
            report["creative_insights"] = creative_report.get("insights_count", 0)
        except Exception as exc:
            logger.debug("Creative association error during consolidate: %s", exc)
            report["creative_error"] = str(exc)

        return report

    def hms_context_inject(self, message: str, tier: str = "auto",
                           **kwargs: Any) -> Dict[str, Any]:
        """
        Tool: Compose full memory-augmented context for an agent turn.

        Args:
            message: current user message
            tier: context tier (auto/32k/128k/256k/1m)

        Returns:
            Context dict with memories, fingerprint, timelines, and summaries.
        """
        mgr = self._get_manager()

        if tier and tier != "auto":
            cfg = Config.get()
            cfg = MemoryManager._apply_tier(cfg, tier)
            mgr = MemoryManager(cfg)

        result = mgr.on_message_received(message)
        return result.get("context", {})

    def hms_handle_conflict(self, old_belief: Dict, new_evidence: Dict,
                            **kwargs: Any) -> Dict[str, Any]:
        """
        Tool: Handle memory conflicts using the MemoryOverwriter.

        Args:
            old_belief: the existing belief to potentially supersede
            new_evidence: the new evidence

        Returns:
            Conflict resolution result.
        """
        if self._overwriter is None:
            self._get_manager()  # ensure init

        return self._overwriter.handle_conflict(old_belief, new_evidence)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release all resources."""
        if self._mgr is not None:
            try:
                self._mgr.close()
            except Exception:
                pass
            self._mgr = None
        logger.info("HMS plugin cleaned up")
