"""
HMS v2 — Unified Memory Manager.

Orchestrates the full cognitive memory pipeline:
  perception → collision → storage → consolidation → compression → fingerprint

Three-layer architecture for infinite context.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import requests

from .perception import PerceptionEngine
from .collision import CollisionEngine
from .context_manager import ContextManager
from .consolidation import ConsolidationEngine
from .forgetting import ForgettingEngine
from .embed_cache import EmbeddingCache

logger = logging.getLogger(__name__)


# ======================================================================
# MemoryAdapter — API isolation layer
# ======================================================================


class MemoryAdapter:
    """
    All calls to memory-lancedb-pro and graph-memory go through here.
    Gateway URL is configurable via config or environment variable.
    """

    GATEWAY_TIMEOUT = 10

    def __init__(self, config: Optional[Dict[str, Any]] = None, tool_impl: Optional[Dict[str, Callable]] = None) -> None:
        self.cfg = config or {}
        self._tools = tool_impl or {}
        
        # Gateway URL: config > env var > default
        self._gateway_url = self.cfg.get(
            "gateway_url",
            os.environ.get("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:3578")
        )

    def _call_gateway_api(self, endpoint: str, payload: Dict[str, Any]) -> Any:
        """Call OpenClaw Gateway internal API via HTTP POST."""
        url = f"{self._gateway_url}{endpoint}"
        resp = requests.post(url, json=payload, timeout=self.GATEWAY_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def store(self, text: str, category: str, importance: int, metadata: str) -> Any:
        try:
            return self._call_gateway_api("/api/tools/memory-store", {
                "text": text,
                "category": category,
                "importance": importance,
                "metadata": metadata,
            })
        except Exception as e:
            fn = self._tools.get("memory_store")
            if fn:
                return fn(text=text, category=category, importance=importance, metadata=metadata)
            logger.debug(f"Gateway store failed, using stub: {e}")
            return {"status": "stubbed", "text": text[:40]}

    def recall(self, query: str, top_k: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            result = self._call_gateway_api("/api/tools/memory-recall", {
                "query": query,
                "top_k": top_k,
                "category": category,
            })
            memories = result if isinstance(result, list) else []
            return self._filter_forgotten(memories)
        except Exception as e:
            fn = self._tools.get("memory_recall")
            if fn:
                memories = fn(query=query, top_k=top_k, category=category) or []
                return self._filter_forgotten(memories)
            logger.debug(f"Gateway recall failed: {e}")
            return []

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
        """Delete a memory. Tries dedicated delete API first, falls back to soft-delete."""
        # Try dedicated delete endpoint
        try:
            result = self._call_gateway_api("/api/tools/memory-forget", {
                "memory_id": memory_id,
            })
            return result
        except Exception as e:
            logger.debug(f"Gateway forget failed: {e}")
        # Try injected tool
        fn = self._tools.get("memory_forget")
        if fn:
            return fn(memory_id=memory_id)
        # Last resort: soft-delete via store with importance=0
        try:
            return self._call_gateway_api("/api/tools/memory-store", {
                "text": "",
                "category": "",
                "importance": 0,
                "metadata": json.dumps({"action": "forget", "memory_id": memory_id}),
            })
        except Exception as e:
            logger.debug(f"Soft-delete forget failed: {e}")
            return {"status": "stubbed", "memory_id": memory_id}

    def graph_record(self, source: str, target: str, relation: str, context: str = "") -> Any:
        try:
            return self._call_gateway_api("/api/tools/gm-record", {
                "source": source,
                "target": target,
                "relation": relation,
                "context": context,
            })
        except Exception as e:
            fn = self._tools.get("gm_record")
            if fn:
                return fn(source=source, target=target, relation=relation, context=context)
            logger.debug(f"Gateway gm-record failed: {e}")
            return {"status": "stubbed"}

    def graph_search(self, query: str, depth: int = 2) -> List[Dict[str, Any]]:
        try:
            result = self._call_gateway_api("/api/tools/gm-search", {
                "query": query,
                "depth": depth,
            })
            return result if isinstance(result, list) else []
        except Exception as e:
            fn = self._tools.get("gm_search")
            if fn:
                return fn(query=query, depth=depth) or []
            logger.debug(f"Gateway gm-search failed: {e}")
            return []


# ======================================================================
# MemoryManager — Main orchestrator
# ======================================================================


class MemoryManager:
    """
    Main HMS v2 orchestrator. Ties all modules together.

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
            os.path.join(os.path.dirname(__file__), "..", "..", "hms", "config.json"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.debug(f"Config load failed for {p}: {e}")
                    continue
        return {}

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
        # 1. Quick perception (heuristic mode for speed)
        perception = self.perception.analyze(user_message, force_llm=False)

        # 2. Retrieve related memories
        retrieved = self.adapter.recall(
            query=user_message,
            top_k=self.cfg.get("retrieval_top_k", 8),
        )

        # 3. Update access counts
        for mem in retrieved:
            mid = mem.get("id", "")
            if mid:
                self.forgetting.update_on_access(mid)

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
        """
        report = {
            "processed": 0,
            "stored": 0,
            "collisions": 0,
            "errors": [],
        }

        entries = self.context.read_pending()
        if not entries:
            return report

        for entry in entries:
            try:
                self._process_single_entry(entry, report)
                report["processed"] += 1
            except Exception as e:
                report["errors"].append(f"entry: {e}")

        self.context.clear_pending()
        return report

    def _process_single_entry(
        self,
        entry: Dict[str, Any],
        report: Dict[str, Any],
    ) -> None:
        """Process one pending entry: perceive → collide → store."""
        user_msg = entry.get("user_message", "")
        reply = entry.get("assistant_reply", "")

        # 1. Deep perception (LLM)
        perception = self.perception.analyze(user_msg, reply, force_llm=True)

        if not perception.get("should_remember", True):
            return

        # 2. Store perception
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
            self.adapter.store(
                text=perception.get("text_for_store", user_msg),
                category=perception.get("category", "fact"),
                importance=perception.get("importance", 5),
                metadata=metadata,
            )
            report["stored"] += 1
        except Exception as e:
            report["errors"].append(f"store: {e}")

        # 3. Collision detection
        retrieved = self.adapter.recall(
            query=user_msg,
            top_k=self.cfg.get("retrieval_top_k", 5),
        )
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

        # 1. Get recent conversations for compression
        pending = self.context.read_pending()
        if pending:
            # Compress in batches
            batch_size = self.cfg.get("compression_window_turns", 10)
            for i in range(0, len(pending), batch_size):
                batch = pending[i : i + batch_size]
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

        # 4. Memory replay
        try:
            all_memories = self.adapter.recall(query="记忆回顾", top_k=100) or []
        except Exception as e:
            logger.debug(f"Memory recall for replay failed: {e}")
            all_memories = []

        if all_memories:
            replay_candidates = self.consolidation.select_for_replay(all_memories, max_count=20)
            for mem in replay_candidates:
                mid = mem.get("id", "")
                # Get related memories
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
                        logger.debug(f"Memory update failed for {mid}: {e}")
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
            all_memories = self.adapter.recall(query="所有记忆", top_k=500) or []
        except Exception as e:
            logger.debug(f"Memory recall for forget failed: {e}")
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


# ======================================================================
# CLI entry point
# ======================================================================


def main():
    """CLI entry point for hook/cron integration."""
    if len(sys.argv) < 2:
        print("Usage: python -m hms.scripts.memory_manager <command> [--tier 32k|128k|256k|1M]")
        print("Commands: received, process_pending, consolidate, forget")
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

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
