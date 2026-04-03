"""
HMS — Unified Memory Manager.

Orchestrates the full cognitive memory pipeline:
  perception → collision → storage → consolidation → compression → fingerprint

Three-layer architecture for infinite context.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import threading

import requests

from .perception import PerceptionEngine
from .collision import CollisionEngine
from .context_manager import ContextManager
from .consolidation import ConsolidationEngine
from .forgetting import ForgettingEngine
from .embed_cache import EmbeddingCache
from .config_loader import Config

logger = logging.getLogger(__name__)


class StoreError(Exception):
    """Raised when memory store operation fails."""
    pass


class RecallError(Exception):
    """Raised when memory recall operation fails."""
    pass


# ======================================================================
# MemoryAdapter — API isolation layer
# ======================================================================


class MemoryAdapter:
    """
    All calls to memory-lancedb-pro and graph-memory go through here.

    Primary path: injected tool functions (OpenClaw native tools).
    Fallback path: HTTP calls to OpenClaw Gateway (for standalone/deployed use).
    """

    GATEWAY_TIMEOUT = 10

    def __init__(self, config: Optional[Dict[str, Any]] = None, tool_impl: Optional[Dict[str, Callable]] = None) -> None:
        # Use unified config if not explicitly provided
        self.cfg = config or Config.get()
        self._tools = tool_impl or {}
        self._session: Optional[requests.Session] = None
        self._session_lock = threading.Lock()

        # Gateway URL: config > env var > default (Config already resolves env vars)
        self._gateway_url = self.cfg.get(
            "gateway_url",
            os.environ.get("HMS_GATEWAY_URL", "http://127.0.0.1:18789")
        )

        # Gateway auth token
        self._gateway_token = self.cfg.get("gateway_token", "")

        # Dedup threshold
        self._dedup_threshold = self.cfg.get("dedup_similarity_threshold", 0.95)
        self._dedup_lock = threading.Lock()

    def close(self) -> None:
        """Close the HTTP session and release connections."""
        with self._session_lock:
            if self._session is not None:
                self._session.close()
                self._session = None

    def __enter__(self) -> "MemoryAdapter":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _get_session(self) -> requests.Session:
        """Lazy-create the HTTP session (only needed for standalone Gateway mode)."""
        if self._session is not None:
            return self._session
        with self._session_lock:
            if self._session is not None:
                return self._session
            s = requests.Session()
            if self._gateway_token:
                s.headers["Authorization"] = f"Bearer {self._gateway_token}"
            self._session = s
            return s

    def _call_gateway_api(self, endpoint: str, payload: Dict[str, Any]) -> Any:
        """Call OpenClaw Gateway internal API via HTTP POST with connection pooling and retry.

        Only used in standalone/deployed mode. Inside OpenClaw, tools are
        preferred and the HTTP path is a fallback.

        The session lock is only used to create the session (via _get_session);
        retries run without holding the lock so concurrent calls are not serialized.
        """
        url = f"{self._gateway_url}{endpoint}"
        session = self._get_session()
        for attempt in range(3):
            try:
                resp = session.post(url, json=payload, timeout=self.GATEWAY_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status in (400, 401, 403, 404, 405):
                    raise
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 5))
                else:
                    raise
            except (requests.exceptions.SSLError, requests.exceptions.InvalidURL,
                    requests.exceptions.InvalidSchema, requests.exceptions.ChunkedEncodingError) as e:
                logger.warning("Request error calling %s: %s", url, e)
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 5))
                else:
                    raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 5))
                else:
                    raise
            except requests.exceptions.RequestException as e:
                logger.warning("Unexpected request error for %s: %s", url, e)
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 5))
                else:
                    raise

    def health_check(self) -> Dict[str, Any]:
        """Verify tool availability or Gateway connectivity."""
        result = {
            "tools_available": bool(self._tools),
            "memory_tool": "memory_store" in self._tools,
            "recall_tool": "memory_recall" in self._tools,
            "graph_tool": "gm_record" in self._tools,
            "gateway_reachable": False,
            "errors": [],
        }
        # If tools are available, we're good — no need to probe Gateway
        if self._tools.get("memory_store"):
            result["memory_api_available"] = True
            result["overall"] = "healthy"
            return result
        # Standalone mode: probe Gateway HTTP
        try:
            session = self._get_session()
            resp = session.get(f"{self._gateway_url}/health", timeout=5)
            if resp.status_code == 200:
                result["gateway_reachable"] = True
                result["memory_api_available"] = True
                result["overall"] = "healthy"
            else:
                result["errors"].append(f"Gateway health returned HTTP {resp.status_code}")
                result["overall"] = "degraded"
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            result["errors"].append(f"Gateway unreachable: {e}")
            result["overall"] = "degraded"
        return result

    def store(self, text: str, category: str, importance: int, metadata: str) -> Any:
        """Store a memory. Tries native tool first, falls back to Gateway HTTP."""
        fn = self._tools.get("memory_store")
        if fn:
            return fn(text=text, category=category, importance=importance, metadata=metadata)
        # Fallback: Gateway HTTP
        try:
            return self._call_gateway_api("/api/tools/memory-store", {
                "text": text,
                "category": category,
                "importance": importance,
                "metadata": metadata,
            })
        except Exception as e:
            logger.warning("Memory store failed (no tool, Gateway fallback error): %s", e)
            raise StoreError(f"store failed: {e}")

    def recall(self, query: str, top_k: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Recall memories. Tries native tool first, falls back to Gateway HTTP."""
        fn = self._tools.get("memory_recall")
        if fn:
            memories = fn(query=query, top_k=top_k, category=category) or []
            return self._filter_forgotten(memories)
        # Fallback: Gateway HTTP
        try:
            result = self._call_gateway_api("/api/tools/memory-recall", {
                "query": query,
                "top_k": top_k,
                "category": category,
            })
            memories = result if isinstance(result, list) else []
            return self._filter_forgotten(memories)
        except Exception as e:
            logger.warning("Memory recall failed (no tool, Gateway fallback error): %s", e)
            raise RecallError(f"recall failed: {e}")

    @staticmethod
    def _filter_forgotten(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out soft-deleted (importance=0) memories."""
        return [m for m in memories if m.get("importance", 1) > 0]

    def update(self, memory_id: str, **kwargs: Any) -> Any:
        fn = self._tools.get("memory_update")
        kwargs["memory_id"] = memory_id
        if fn:
            return fn(**kwargs)
        return {"status": "stubbed"}

    def forget(self, memory_id: str) -> Any:
        """Delete a memory. Tries native tool first, falls back to Gateway HTTP."""
        # Try injected tool first (native OpenClaw path)
        fn = self._tools.get("memory_forget")
        if fn:
            return fn(memory_id=memory_id)
        # Fallback: Gateway HTTP
        try:
            result = self._call_gateway_api("/api/tools/memory-forget", {
                "memory_id": memory_id,
            })
            return result
        except Exception as e:
            logger.debug("Gateway forget failed: %s", e)
        # Last resort: soft-delete via update with importance=0
        try:
            return self._call_gateway_api("/api/tools/memory-update", {
                "memory_id": memory_id,
                "importance": 0,
                "metadata": json.dumps({"action": "soft_delete", "original_id": memory_id}),
            })
        except Exception as e:
            logger.debug("Soft-delete forget failed: %s", e)
            return {"status": "stubbed", "memory_id": memory_id}

    def graph_record(self, source: str, target: str, relation: str, context: str = "") -> Any:
        """Record a graph edge. Tries native tool first, falls back to Gateway HTTP."""
        fn = self._tools.get("gm_record")
        if fn:
            return fn(source=source, target=target, relation=relation, context=context)
        try:
            return self._call_gateway_api("/api/tools/gm-record", {
                "source": source,
                "target": target,
                "relation": relation,
                "context": context,
            })
        except Exception as e:
            logger.warning("Graph record failed (no tool, Gateway fallback error): %s", e)
            raise StoreError(f"graph_record failed: {e}")

    def graph_search(self, query: str, depth: int = 2) -> List[Dict[str, Any]]:
        """Search the knowledge graph. Tries native tool first, falls back to Gateway HTTP."""
        fn = self._tools.get("gm_search")
        if fn:
            return fn(query=query, depth=depth) or []
        try:
            result = self._call_gateway_api("/api/tools/gm-search", {
                "query": query,
                "depth": depth,
            })
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.warning("Graph search failed (no tool, Gateway fallback error): %s", e)
            raise RecallError(f"graph_search failed: {e}")

    def store_with_dedup(
        self, text: str, category: str, importance: int, metadata: str,
        embed_cache: Optional[EmbeddingCache] = None,
    ) -> Any:
        """Store memory with deduplication using embedding similarity.

        Dedup is best-effort — if recall is unavailable (no tools, no Gateway),
        falls back to plain store without raising.
        """
        with self._dedup_lock:
            if embed_cache is not None:
                try:
                    similar = self.recall(query=text, top_k=3)
                except (RecallError, StoreError):
                    logger.debug("Recall unavailable for dedup, storing without dedup check")
                    similar = []
                for mem in similar:
                    existing_text = mem.get("text", "")
                    if not existing_text:
                        continue
                    sim = embed_cache.similarity(text, existing_text)
                    if sim >= self._dedup_threshold:
                        # Update existing memory instead of creating new one
                        logger.info(
                            "Dedup: merging into existing memory %s (similarity=%.3f)",
                            mem.get("id", ""), sim,
                        )
                        return self.update(mem["id"], importance=max(mem.get("importance", 0), importance))

            return self.store(text, category, importance, metadata)


# ======================================================================
# MemoryManager — Main orchestrator
# ======================================================================


class MemoryManager:
    """
    HMS orchestrator. Ties all modules together.

    Entry points:
      - on_message_received(): sync perception + context injection
      - on_message_sent(): async queue write
      - process_pending(): batch LLM analysis
      - consolidate(): daily consolidation
      - forget(): weekly forgetting
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tool_impl: Optional[Dict[str, Callable]] = None,
        context_tier: str = "",
    ) -> None:
        self.cfg = config or self._load_config()

        # Apply context tier if specified
        if context_tier:
            self.cfg = self._apply_tier(self.cfg, context_tier)

        # Init adapter with config for gateway_url
        self.adapter = MemoryAdapter(self.cfg, tool_impl)

        # Init sub-modules
        self.perception = PerceptionEngine(self.cfg)
        self.collision_engine = CollisionEngine(self.cfg)
        self.forgetting = ForgettingEngine({
            **self.cfg,
            "decay_cache_path": os.path.join(
                self.cfg.get("cache_dir", "cache"), "decay_state.json"
            ),
        })
        self.consolidation = ConsolidationEngine(self.cfg)
        self.context = ContextManager({
            **self.cfg,
            "pending_path": os.path.join(
                self.cfg.get("cache_dir", "cache"), "pending_processing.jsonl"
            ),
        })

        # Embedding cache for pre-filtering (reduces LLM calls by 60-70%)
        cache_dir = self.cfg.get("cache_dir", "cache")
        self.embed_cache = EmbeddingCache({"cache_dir": cache_dir})
        self.collision_engine.set_embed_cache(self.embed_cache)
        self.consolidation.set_embed_cache(self.embed_cache)

    @staticmethod
    def _apply_tier(cfg: Dict[str, Any], tier: str) -> Dict[str, Any]:
        """Merge a context tier config into the base config."""
        tiers = cfg.get("context_tiers", {})
        tier_cfg = tiers.get(tier, {})
        if not tier_cfg:
            return cfg
        merged = dict(cfg)
        for key, val in tier_cfg.items():
            if key == "context_budget" and isinstance(val, dict):
                merged["context_budget"] = {**merged.get("context_budget", {}), **val}
            else:
                merged[key] = val
        return merged

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        """Try to load config.json from common locations."""
        candidates = [
            "config.json",
            os.path.join(os.path.dirname(__file__), "..", "config.json"),
            os.path.join(os.path.dirname(__file__), "config.json"),
            os.path.expanduser("~/.openclaw/hms/config.json"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.debug("Config load failed for %s: %s", p, e)
                    continue
        return {}

    def health_check(self) -> Dict[str, Any]:
        """Run full system health check at startup."""
        result = {
            "gateway": self.adapter.health_check(),
            "llm": self.perception.llm.health_check(),
            "embedding": self.embed_cache.get_stats(),
            "overall": "healthy",
        }
        # Check if any critical component is down
        if not result["gateway"]["gateway_reachable"]:
            result["overall"] = "degraded"
            logger.warning("Health check: Gateway unreachable, system will use stubs")
        if not result["llm"]["chat_api_available"]:
            result["overall"] = "degraded"
            logger.warning("Health check: LLM chat API unavailable, falling back to heuristics")
        return result

    # ==================================================================
    # 1. On message received (SYNC)
    # ==================================================================

    def on_message_received(
        self,
        user_message: str,
        session_id: str = "",
    ) -> Dict[str, Any]:
        """
        Sync handler for incoming messages.
        Does lightweight perception + context injection.

        Returns context to inject into the agent's turn.
        """
        MAX_MESSAGE_LENGTH = 4096
        if len(user_message) > MAX_MESSAGE_LENGTH:
            logger.warning("Message too long (%d chars), truncating to %d", len(user_message), MAX_MESSAGE_LENGTH)
            user_message = user_message[:MAX_MESSAGE_LENGTH] + "..."

        # 1. Quick perception (heuristic only — no LLM for sync path)
        # assistant_reply is intentionally empty here; reply is not yet generated.
        # Full analysis with assistant_reply happens later in _process_single_entry.
        perception = self.perception.analyze(user_message, assistant_reply="", force_heuristic=True)

        # 2. Retrieve related memories
        try:
            retrieved = self.adapter.recall(
                query=user_message,
                top_k=self.cfg.get("retrieval_top_k", 8),
            )
        except RecallError:
            logger.warning("Recall failed in on_message_received, returning empty")
            retrieved = []

        # 3. Update access counts
        for mem in retrieved:
            mid = mem.get("id", "")
            if mid:
                self.forgetting.update_on_access(mid)
        self.forgetting.flush()

        # 4. Compose context
        recent_turns = self.context.read_pending()[-self.cfg.get("working_memory_recent_turns", 15):]
        context_result = self.context.compose_context(
            system_prompt="",
            injected_memories=retrieved,
            recent_turns=[
                {"user": t.get("user_message", ""), "assistant": t.get("assistant_reply", "")}
                for t in recent_turns
            ],
        )

        return {
            "perception": perception,
            "retrieved_memories": retrieved,
            "context": context_result,
        }

    # ==================================================================
    # 2. On message sent (ASYNC)
    # ==================================================================

    def on_message_sent(
        self,
        user_message: str,
        assistant_reply: str,
        session_id: str = "",
    ) -> None:
        """
        Async handler after assistant reply.
        Writes to pending queue for batch processing.
        """
        self.context.enqueue(user_message, assistant_reply, session_id)

    # ==================================================================
    # 3. Process pending (cron: every minute or before compaction)
    # ==================================================================

    def process_pending(self) -> Dict[str, Any]:
        """
        Batch-process all pending entries.
        Each entry: perception → collision → store.
        Processes at most 50 entries per call to prevent resource exhaustion.
        Remaining entries will be picked up on the next cron run.
        """
        report = {
            "processed": 0,
            "stored": 0,
            "collisions": 0,
            "dedup_merged": 0,
            "errors": [],
        }

        entries = self.context.pop_all_pending()
        if not entries:
            return report

        # Limit batch size to prevent resource exhaustion
        max_batch = self.cfg.get("process_pending_max_batch", 50)
        if len(entries) > max_batch:
            logger.warning(
                "Pending batch size %d exceeds limit %d. Processing first %d, "
                "remaining %d will be processed on next run.",
                len(entries), max_batch, max_batch, len(entries) - max_batch,
            )
            # Push deferred entries to the FRONT of the queue (prevent starvation)
            excess = entries[max_batch:]
            entries = entries[:max_batch]
            # Write excess first, then existing pending items — under file_lock
            from .file_utils import file_lock
            pending_path = Path(self.context._cache_dir) / 'pending_processing.jsonl'
            with file_lock(str(pending_path)):
                existing_lines = []
                if pending_path.exists():
                    with open(pending_path, 'r', encoding='utf-8') as f:
                        existing_lines = [l for l in f if l.strip()]
                with open(pending_path, 'w', encoding='utf-8') as f:
                    for entry in excess:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    for line in existing_lines:
                        f.write(line)

        for entry in entries:
            try:
                self._process_single_entry(entry, report)
                report["processed"] += 1
            except Exception as e:
                report["errors"].append(f"entry {entry.get('timestamp', '?')}: {e}")

        return report

    def _process_single_entry(
        self,
        entry: Dict[str, Any],
        report: Dict[str, Any],
    ) -> None:
        """Process one pending entry: perceive → collide → store."""
        user_msg = entry.get("user_message", "")
        reply = entry.get("assistant_reply", "")

        # 1. Perception: heuristic-only for async path (LLM calls gated by importance score)
        perception = self.perception.analyze(user_msg, reply, force_heuristic=True)
        if not perception.get("should_remember", True):
            return

        # 2. Store perception with dedup
        metadata = json.dumps({
            "belief_strength": "uncertain",
            "belief_confidence": 0.5,
            "emotional_valence": perception.get("emotion", {}).get("valence", 0.0),
            "emotional_arousal": perception.get("emotion", {}).get("arousal", 0.0),
            "primary_emotion": perception.get("emotion", {}).get("primary_emotion", "neutral"),
            "memory_type": "episodic",
            "source": "direct_experience",
            "entities": [
                e.get("name", str(e)) if isinstance(e, dict) else str(e)
                for e in perception.get("entities", [])
            ],
            "topics": perception.get("topics", []),
        }, ensure_ascii=False)

        try:
            self.adapter.store_with_dedup(
                text=perception.get("text_for_store", user_msg),
                category=perception.get("category", "fact"),
                importance=perception.get("importance", 5),
                metadata=metadata,
                embed_cache=self.embed_cache,
            )
            report["stored"] += 1
        except Exception as e:
            report["errors"].append(f"store: {e}")

        # 3. Collision detection (best-effort: skip if recall unavailable)
        try:
            retrieved = self.adapter.recall(
                query=user_msg,
                top_k=self.cfg.get("retrieval_top_k", 5),
            )
        except RecallError:
            retrieved = []
        if retrieved:
            collision_result = self.collision_engine.collide(perception, retrieved)
            exec_report = self.collision_engine.execute_results(
                collision_result,
                memory_store_func=lambda **kw: self.adapter.store(**kw),
                gm_record_func=lambda **kw: self.adapter.graph_record(**kw),
            )
            report["collisions"] += 1

            # Update reinforcement counts
            for r in collision_result.get("reinforcements", []):
                mid = r.get("existing_id", "")
                if mid:
                    self.forgetting.update_on_reinforce(mid)
            self.forgetting.flush()

    # ==================================================================
    # 4. Consolidate (cron: daily 3AM)
    # ==================================================================

    def consolidate(self) -> Dict[str, Any]:
        """
        Daily consolidation:
          1. Select memories for replay
          2. Compress recent conversations
          3. Update topic timelines
          4. Update cognitive fingerprint
          5. Discover relations
        """
        report = {
            "replayed": 0,
            "compressed": 0,
            "timeline_updates": 0,
            "fingerprint_updated": False,
            "relations_found": 0,
            "errors": [],
        }

        # 0. Flush pending first to avoid conflict with concurrent process_pending()
        self.process_pending()

        # 1. Single recall: fetch all memories ONCE, reuse across replay/relations
        try:
            all_memories = self.adapter.recall(query="", top_k=100) or []
        except Exception as e:
            logger.debug("Memory recall for consolidation failed: %s", e)
            all_memories = []

        # 1. Get recent conversations for compression — use stored summaries, not pending
        compressed_list = self.context.get_compressed_summaries(since_hours=24)
        if compressed_list:
            # Compress in batches
            batch_size = self.cfg.get("compression_window_turns", 10)
            for i in range(0, len(compressed_list), batch_size):
                batch = compressed_list[i : i + batch_size]
                conversations = [
                    {"user": e.get("user_message", ""), "assistant": e.get("assistant_reply", "")}
                    for e in batch
                ]
                compressed = self.consolidation.compress_conversations(
                    conversations,
                    self.context.get_fingerprint(),
                    max_summary_tokens=self.cfg.get("compression_max_summary_tokens", 500),
                )
                if compressed:
                    self.context.add_compressed_summary(compressed)
                    report["compressed"] += 1

        # 2. Build timeline entries from compressed summaries
        recent_summaries = self.context.get_compressed_summaries(since_hours=48)
        timeline_entries = self.consolidation.build_timeline_entries(recent_summaries)
        if timeline_entries:
            self.context.update_timelines(timeline_entries)
            report["timeline_updates"] = len(timeline_entries)

        # 3. Update cognitive fingerprint
        if recent_summaries:
            fp_result = self.consolidation.update_fingerprint(
                self.context.get_fingerprint(),
                recent_summaries,
            )
            if fp_result:
                self.context.update_fingerprint(fp_result)
                report["fingerprint_updated"] = True

        # 4. Memory replay (reuses all_memories from step 0)
        if all_memories:
            replay_candidates = self.consolidation.select_for_replay(all_memories, max_count=20)
            for mem in replay_candidates:
                mid = mem.get("id", "")
                # Get related memories from pre-fetched all_memories (avoid recall API calls)
                mem_text = mem.get("text", "")[:100]
                related = [
                    m for m in all_memories
                    if m.get("id", "") != mid and mem_text[:20] in m.get("text", "")
                ][:5]
                if not related:
                    related = self.adapter.recall(
                        query=mem.get("text", "")[:100], top_k=5
                    )
                result = self.consolidation.replay_memory(mem, related)
                if result.get("issues"):
                    try:
                        self.adapter.update(
                            memory_id=mid,
                            importance_delta=result.get("importance_adjustment", 0),
                        )
                    except Exception as e:
                        logger.debug("Memory update failed for %s: %s", mid, e)
                    report["replayed"] += 1

        # 5. Relation discovery
        if all_memories:
            relations = self.consolidation.discover_relations(all_memories[:50])
            if relations:
                count = self.consolidation.apply_relations(
                    relations,
                    gm_record_func=lambda **kw: self.adapter.graph_record(**kw),
                )
                report["relations_found"] = count

        # 6. Decay state consistency check
        if all_memories:
            sync_report = self.forgetting.sync_consistency(all_memories)
            report["decay_sync"] = sync_report

        return report

    # ==================================================================
    # 5. Forget (cron: weekly Sunday 4AM)
    # ==================================================================

    def forget(self) -> Dict[str, Any]:
        """
        Weekly forgetting pass.
        Evaluate all memories and forget weak ones.
        """
        try:
            all_memories = self.adapter.recall(query="", top_k=500) or []
        except Exception as e:
            logger.debug("Memory recall for forget failed: %s", e)
            all_memories = []

        evaluation = self.forgetting.evaluate_all(all_memories)

        deleted = self.forgetting.execute_forgetting(
            evaluation["to_forget"],
            memory_forget_func=lambda mid: self.adapter.forget(mid),
        )

        return {
            "evaluated": evaluation["report"]["total_evaluated"],
            "forgotten": deleted,
            "kept": len(evaluation["to_keep"]),
            "forget_ratio": evaluation["report"]["forget_ratio"],
        }

    def close(self) -> None:
        """Close all resources (HTTP sessions, file descriptors)."""
        try:
            self.adapter.close()
        except Exception:
            pass
        try:
            self.perception.llm.close()
        except Exception:
            pass
        try:
            self.collision_engine.llm.close()
        except Exception:
            pass
        try:
            self.consolidation.llm.close()
        except Exception:
            pass
        try:
            from .file_utils import close_all_lock_fds
            close_all_lock_fds()
        except Exception:
            pass

    def __enter__(self) -> "MemoryManager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ======================================================================
# CLI entry point
# ======================================================================


def main():
    """CLI entry point for hook/cron integration."""
    if len(sys.argv) < 2:
        print("Usage: python -m hms <command> [--tier 32k|128k|256k|1M]")
        print("Commands: received, process_pending, consolidate, forget, health")
        sys.exit(1)

    command = sys.argv[1]

    # Parse optional tier
    tier = ""
    args = sys.argv[2:]
    if "--tier" in args:
        idx = args.index("--tier")
        if idx + 1 < len(args):
            tier = args[idx + 1]
            args = args[:idx] + args[idx + 2:]

    mgr = MemoryManager(context_tier=tier)

    if command == "received":
        # Read message from stdin or args
        if args:
            msg = " ".join(args)
        else:
            msg = sys.stdin.read().strip()
        result = mgr.on_message_received(msg)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif command == "process_pending":
        result = mgr.process_pending()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif command == "consolidate":
        result = mgr.consolidate()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif command == "forget":
        result = mgr.forget()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif command == "health":
        result = mgr.health_check()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
