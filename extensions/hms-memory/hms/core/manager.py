"""
MemoryManager v5 — HMS Unified Scheduler.

Adapted from hms-v4 for v5:
  - Config-driven (no Config singleton, no .env)
  - SQLite adapter with Ollama embeddings (no LanceDB)
  - Shared LLMAnalyzer instance
  - Thread-safe pending queue
"""

from __future__ import annotations
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

from hms.utils.llm import LLMAnalyzer
from hms.core.adapter import StorageAdapter
from hms.engines.perception import PerceptionEngine
from hms.engines.collision import CollisionEngine
from hms.engines.forgetting import ForgettingEngine
from hms.engines.consolidation import ConsolidationEngine
from hms.engines.embed import EmbeddingCache

logger = logging.getLogger("hms.manager")


class MemoryManager:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        cache_dir = config.get("cache_dir", config.get("dataDir", "data"))
        os.makedirs(cache_dir, exist_ok=True)

        self.llm = LLMAnalyzer(config)

        self.adapter = StorageAdapter(config)

        self.perception = PerceptionEngine(config, llm=self.llm)
        self.collision_engine = CollisionEngine(config, llm=self.llm)
        self.forgetting = ForgettingEngine({
            **config,
            "decay_cache_path": os.path.join(cache_dir, "decay_state.json"),
        })
        self.consolidation = ConsolidationEngine(config, llm=self.llm)

        self.embed_cache = EmbeddingCache({
            "cache_dir": cache_dir,
            **config,
        })

        self.collision_engine.set_embed_cache(self.embed_cache)
        self.consolidation.set_embed_cache(self.embed_cache)

        self._pending_path = os.path.join(cache_dir, "pending.jsonl")
        self._pending_lock = threading.Lock()

    def on_message_received(self, user_message: str) -> Dict[str, Any]:
        MAX_LEN = 4096
        if len(user_message) > MAX_LEN:
            user_message = user_message[:MAX_LEN] + "..."

        perception = self.perception.analyze(user_message, "", force_heuristic=True)

        try:
            top_k = self.cfg.get("retrievalTopK", 30)
            retrieved = self.adapter.recall(query=user_message, top_k=top_k)
        except Exception:
            logger.warning("Recall failed")
            retrieved = []

        for mem in retrieved:
            mid = mem.get("id", "")
            if mid:
                self.forgetting.update_on_access(mid)
                self.adapter.increment_access(mid)
        self.forgetting.flush()

        context_text = self._compose_context(perception, retrieved)

        return {
            "perception": perception,
            "retrieved_memories": retrieved[:8],
            "context": context_text,
        }

    def on_message_sent(self, user_message: str, assistant_reply: str) -> None:
        with self._pending_lock:
            entry = {
                "user_message": user_message,
                "assistant_reply": assistant_reply,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(self._pending_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def consolidate(self) -> Dict[str, Any]:
        report = {"processed": 0, "stored": 0, "compressed": 0, "errors": []}
        self._process_pending(report)
        try:
            all_memories = self.adapter.get_all_memories(limit=100)
        except Exception:
            all_memories = []
        try:
            self.consolidation.run(all_memories, report)
        except Exception as e:
            report["errors"].append(f"consolidation: {e}")
        return report

    def _process_pending(self, report: Dict[str, Any]) -> None:
        entries = []
        with self._pending_lock:
            if not os.path.exists(self._pending_path):
                return
            try:
                with open(self._pending_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                open(self._pending_path, "w").close()
            except IOError as e:
                report["errors"].append(f"pending read: {e}")
                return

        if not entries:
            return

        max_batch = self.cfg.get("processPendingMaxBatch", 50)
        for entry in entries[:max_batch]:
            try:
                self._process_single_entry(entry, report)
                report["processed"] += 1
            except Exception as e:
                report["errors"].append(str(e))

    def _process_single_entry(self, entry: dict, report: dict) -> None:
        user_msg = entry.get("user_message", "")
        reply = entry.get("assistant_reply", "")

        perception = self.perception.analyze(user_msg, reply, force_heuristic=True)
        if not perception.get("should_remember", True):
            return

        metadata = json.dumps({
            "belief_strength": "uncertain",
            "belief_confidence": 0.5,
            "emotional_valence": perception.get("emotion", {}).get("valence", 0.0),
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

    def _compose_context(self, perception: dict, memories: list) -> str:
        if not memories:
            return ""
        lines = ["=== 相关记忆 ==="]
        for i, mem in enumerate(memories[:5], 1):
            text = mem.get("text", "")[:200]
            importance = mem.get("importance", 5)
            lines.append(f"记忆 [{i}] (重要性: {importance}): {text}")
        lines.append("=== 记忆结束 ===")
        return "\n".join(lines)

    def _count_pending(self) -> int:
        try:
            with open(self._pending_path, "r") as f:
                return sum(1 for line in f if line.strip())
        except FileNotFoundError:
            return 0

    def health_check(self) -> Dict[str, Any]:
        return {
            "adapter": self.adapter.health_check(),
            "pending_queue": self._count_pending(),
            "embedding_backend": self.cfg.get("embeddingBackend", "ollama"),
            "status": "healthy",
        }

    def close(self) -> None:
        self.adapter.close()
        self.forgetting.flush()

    @staticmethod
    def _apply_tier(cfg: Dict[str, Any], tier: str) -> Dict[str, Any]:
        tiers = cfg.get("context_tiers", {})
        tier_cfg = tiers.get(tier, {})
        if not tier_cfg:
            return cfg
        merged = dict(cfg)
        merged.update(tier_cfg)
        return merged
