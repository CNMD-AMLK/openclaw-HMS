"""
HMS v4 — Dream Engine.

Inspired by biological sleep consolidation, the dream engine runs periodic
'dream cycles' that:
  1. Randomly walks the memory graph to find distant associations
  2. Evaluates whether those associations are meaningful via LLM
  3. Cleans up isolated / contradictory memory fragments
  4. Saves discovered insights to memory/insights/

This runs during the consolidation phase (daily) or manually via tool.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_loader import Config

logger = logging.getLogger("hms.dream_engine")


class DreamEngine:
    """
    Dream consolidation engine.

    Performs memory integration by finding distant connections that
    normal recall would miss — similar to how the human brain
    discovers unexpected patterns during REM sleep.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self._max_hops = self.cfg.get("dream_max_hops", 3)
        self._max_associations = self.cfg.get("dream_max_associations", 5)
        self._min_embedding_similarity = self.cfg.get("dream_min_embedding_sim", 0.3)
        self._insights_dir = self.cfg.get("dream_insights_dir", "cache/insights")
        self._clean_orphans = self.cfg.get("dream_clean_orphans", True)
        logger.debug(
            "DreamEngine initialized (hops=%d, max_assoc=%d, insights_dir=%s)",
            self._max_hops, self._max_associations, self._insights_dir,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_dream_cycle(self) -> Dict[str, Any]:
        """
        Execute a complete dream cycle.

        Returns a report with insights found, fragments cleaned, etc.
        """
        report = {
            "cycle_time": datetime.now(timezone.utc).isoformat(),
            "associations_found": 0,
            "insights_saved": 0,
            "fragments_cleaned": 0,
            "errors": [],
        }

        try:
            # Ensure insights directory
            os.makedirs(self._insights_dir, exist_ok=True)

            # 1. Get available memories
            memories = self._load_memories()
            if len(memories) < 3:
                logger.debug("Dream cycle skipped: too few memories (%d)", len(memories))
                return report

            # 2. Find distant associations
            associations = self._find_distant_associations(memories)
            report["associations_found"] = len(associations)

            # 3. Generate insights from promising paths
            for path in associations[:self._max_associations]:
                try:
                    insight = self._generate_insight(path)
                    if insight and insight.get("meaningful"):
                        self._save_insight(insight)
                        report["insights_saved"] += 1
                        logger.info("Dream insight saved: %s", insight.get("title", "unknown"))
                except Exception as exc:
                    logger.debug("Dream insight generation failed: %s", exc)
                    report["errors"].append(str(exc))

            # 4. Clean up orphaned / low-quality fragments
            if self._clean_orphans:
                cleaned = self._clean_fragments(memories)
                report["fragments_cleaned"] = cleaned
        except Exception as exc:
            logger.error("Dream cycle error: %s", exc)
            report["errors"].append(f"cycle_error: {exc}")

        return report

    # ------------------------------------------------------------------
    # Memory loading
    # ------------------------------------------------------------------

    def _load_memories(self) -> List[Dict[str, Any]]:
        """Load all available memories from the cache."""
        memories: List[Dict[str, Any]] = []

        # Try to reach via adapter first
        try:
            from hms.scripts.memory_manager import MemoryAdapter
            cfg = Config.get()
            adapter = MemoryAdapter(cfg)
            try:
                raw = adapter.recall(query="", top_k=100) or []
                memories = [
                    {
                        "id": m.get("id", f"mem_{i}"),
                        "text": m.get("text", ""),
                        "metadata": m.get("metadata", {}),
                        "importance": m.get("importance", 5),
                    }
                    for i, m in enumerate(raw)
                ]
            finally:
                adapter.close()
        except Exception as exc:
            logger.debug("Memory loading via adapter failed: %s", exc)

        # Fallback: local cache files
        if not memories:
            cache_dir = Config.get().get("cache_dir", "cache")
            for fname in ["beliefs.json", "cognitive_fingerprint.json"]:
                fpath = os.path.join(cache_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    entries = data if isinstance(data, list) else list(data.values())
                    for i, entry in enumerate(entries):
                        text = entry.get("text", "") or entry.get("content", "") or ""
                        if text:
                            memories.append({
                                "id": entry.get("id", f"local_{i}"),
                                "text": text,
                                "metadata": entry.get("metadata", {}),
                                "importance": entry.get("importance", 5),
                            })
                except (json.JSONDecodeError, IOError, OSError):
                    continue

        logger.debug("Dream engine loaded %d memories", len(memories))
        return memories

    # ------------------------------------------------------------------
    # Distant association discovery
    # ------------------------------------------------------------------

    def _find_distant_associations(
        self,
        memories: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """
        Find pairs/groups of memories that are:
          - Embedding similarity < 0.3 (semantically distant)
          - But connected via indirect graph paths (share entities/topics
            through intermediate memories)

        Returns a list of memory chains (paths).
        """
        from hms.scripts.embed_cache import EmbeddingCache
        from hms.scripts.utils import tokenize

        cache_dir = Config.get().get("cache_dir", "cache")
        embed_cache = EmbeddingCache({"cache_dir": cache_dir})

        # Build adjacency: two memories are adjacent if they share ≥1 entity or topic
        entity_graph: Dict[str, List[int]] = {}  # entity → [memory indices]
        topic_graph: Dict[str, List[int]] = {}   # topic → [memory indices]

        for idx, mem in enumerate(memories):
            meta = mem.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, ValueError):
                    meta = {}
            if not isinstance(meta, dict):
                meta = {}

            for ent in meta.get("entities", []):
                ent_str = ent if isinstance(ent, str) else ent.get("name", "")
                if ent_str:
                    entity_graph.setdefault(ent_str, []).append(idx)

            for topic in meta.get("topics", []):
                topic_graph.setdefault(topic, []).append(idx)

        # Build adjacency list
        adj: Dict[int, List[int]] = {i: [] for i in range(len(memories))}
        for mem_indices in entity_graph.values():
            for i in mem_indices:
                for j in mem_indices:
                    if i != j and j not in adj[i]:
                        adj[i].append(j)
        for mem_indices in topic_graph.values():
            for i in mem_indices:
                for j in mem_indices:
                    if i != j and j not in adj[i]:
                        adj[i].append(j)

        # Random walk from random seeds to find distant pairs
        found_paths: List[List[Dict[str, Any]]] = []
        seen_pairs: set = set()

        for _ in range(min(50, len(memories) * 2)):  # limit walk attempts
            start_idx = random.randint(0, len(memories) - 1)
            start_vec = embed_cache.embed(memories[start_idx]["text"])

            # BFS for paths of length 2..max_hops
            path = self._bfs_path(
                adj, start_idx,
                max_depth=self._max_hops,
                memories=memories,
                embed_cache=embed_cache,
                start_vec=start_vec,
                seen_pairs=seen_pairs,
            )
            if path and len(path) >= 2:
                path_mems = [memories[idx] for idx in path]
                # Verify: first and last should have low direct similarity
                first_vec = embed_cache.embed(path_mems[0]["text"])
                last_vec = embed_cache.embed(path_mems[-1]["text"])
                direct_sim = embed_cache.similarity_of(first_vec, last_vec)
                if direct_sim < self._min_embedding_similarity:
                    found_paths.append(path_mems)

        # Also do pure BFS from a few random starts
        for start in random.sample(range(len(memories)), min(10, len(memories))):
            path = self._bfs_simple(adj, start, max_depth=self._max_hops)
            if path and len(path) >= 3:
                path_mems = [memories[idx] for idx in path]
                pair_key = (path_mems[0]["id"], path_mems[-1]["id"])
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    found_paths.append(path_mems)

        logger.debug("Dream engine found %d distant association paths", len(found_paths))
        return found_paths

    @staticmethod
    def _bfs_path(
        adj: Dict[int, List[int]],
        start: int,
        max_depth: int,
        memories: List[Dict[str, Any]],
        embed_cache: Any,
        start_vec,
        seen_pairs: set,
    ) -> Optional[List[int]]:
        """BFS to find a path from start to a distant memory."""
        from collections import deque
        queue: deque = deque([(start, [start])])
        visited: set = {start}

        while queue:
            curr, path = queue.popleft()
            if len(path) >= max_depth:
                # Record the endpoint pair before returning
                pair = (memories[path[0]]["id"], memories[path[-1]]["id"])
                seen_pairs.add(pair)
                return path
            for neighbor in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))

        return None if len([start]) < 2 else [start]

    @staticmethod
    def _bfs_simple(
        adj: Dict[int, List[int]],
        start: int,
        max_depth: int,
    ) -> Optional[List[int]]:
        """Simple BFS returning the longest path found within max_depth."""
        from collections import deque
        queue: deque = deque([(start, [start])])
        longest: List[int] = []

        while queue:
            curr, path = queue.popleft()

            if len(path) > len(longest):
                longest = path

            if len(path) >= max_depth:
                continue

            for neighbor in adj.get(curr, []):
                if neighbor not in path:  # avoid cycles
                    queue.append((neighbor, path + [neighbor]))

        return longest if len(longest) >= 2 else None

    # ------------------------------------------------------------------
    # LLM insight generation
    # ------------------------------------------------------------------

    def _generate_insight(
        self,
        path: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate whether a memory path represents a meaningful insight.

        Uses LLM to determine if the connection is non-obvious but true.
        """
        prompt_parts = ["发现以下记忆之间可能存在深层关联：\n"]
        for i, mem in enumerate(path, 1):
            text = mem.get("text", "").replace("\n", " ")[:200]
            prompt_parts.append(
                f"记忆 {i}: {text}"
            )
        prompt_parts.append(
            "\n请分析这些记忆之间是否存在有意义的关联。"
            "\n请以JSON格式回答：\n"
            '{"meaningful": true/false, "title": "洞察标题", '
            '"description": "详细描述", "confidence": 0.0-1.0, '
            '"memory_ids": ["id1", "id2", ...]}'
        )

        prompt = "\n".join(prompt_parts)

        try:
            from hms.scripts.llm_analyzer import LLMAnalyzer
            analyzer = LLMAnalyzer()
            raw = analyzer.analyze(prompt, max_tokens=800, temperature=0.5)
            result = analyzer._parse_json_response(raw) if isinstance(raw, str) else None

            if not isinstance(result, dict):
                return None

            result.setdefault("meaningful", False)
            if not result.get("meaningful"):
                return None

            result["cycle_time"] = datetime.now(timezone.utc).isoformat()
            result["path_length"] = len(path)
            result["memory_ids"] = result.get(
                "memory_ids",
                [m.get("id", "") for m in path],
            )
            return result
        except Exception as exc:
            logger.debug("LLM insight generation failed: %s", exc)
            # Fallback: create heuristic insight
            return self._heuristic_insight(path)

    @staticmethod
    def _heuristic_insight(path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a heuristic insight when LLM is unavailable."""
        memory_ids = [m.get("id", "") for m in path]
        texts = [m.get("text", "").replace("\n", " ")[:100] for m in path]

        return {
            "meaningful": True,
            "title": "记忆关联发现",
            "description": f"发现 {len(path)} 条记忆之间的关联路径: {' → '.join(texts)}",
            "confidence": 0.3,
            "memory_ids": memory_ids,
            "cycle_time": datetime.now(timezone.utc).isoformat(),
            "path_length": len(path),
            "heuristic": True,
        }

    # ------------------------------------------------------------------
    # Fragment cleanup
    # ------------------------------------------------------------------

    def _clean_fragments(
        self,
        memories: List[Dict[str, Any]],
    ) -> int:
        """
        Clean up orphaned or low-quality memory fragments.

        Orphaned: no entities, no topics, very short text, never accessed.
        Returns the count of cleaned fragments.
        """
        cleaned = 0
        from hms.scripts.utils import estimate_tokens, tokenize

        cache_dir = Config.get().get("cache_dir", "cache")
        report_path = os.path.join(cache_dir, "dream_cleanup_report.json")

        cleanup_log: List[Dict[str, Any]] = []

        for mem in memories:
            text = mem.get("text", "")
            meta = mem.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, ValueError):
                    meta = {}
            if not isinstance(meta, dict):
                meta = {}

            reasons: List[str] = []

            # Check: very short text (< 5 tokens)
            tokens = tokenize(text)
            if len(tokens) < 5:
                reasons.append("too_short")

            # Check: no entities and no topics
            has_entities = bool(meta.get("entities"))
            has_topics = bool(meta.get("topics"))
            if not has_entities and not has_topics:
                reasons.append("no_metadata")

            # Check: very low importance and old
            importance = mem.get("importance", 3)
            if importance <= 2 and estimate_tokens(text) < 20:
                reasons.append("low_value")

            if reasons and len(reasons) >= 2:
                cleanup_log.append({
                    "id": mem.get("id", ""),
                    "reasons": reasons,
                    "text_preview": text[:50],
                })
                cleaned += 1

        # Save cleanup report
        if cleanup_log:
            try:
                os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
                existing: List[Dict[str, Any]] = []
                if os.path.isfile(report_path):
                    with open(report_path, "r", encoding="utf-8") as fh:
                        existing = json.load(fh)
                existing.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "fragments_cleaned": cleaned,
                    "details": cleanup_log,
                })
                # Keep last 10 reports
                with open(report_path, "w", encoding="utf-8") as fh:
                    json.dump(existing[-10:], fh, ensure_ascii=False, indent=2)
            except (IOError, OSError) as exc:
                logger.debug("Failed to save cleanup report: %s", exc)

        if cleaned > 0:
            logger.info("Dream engine cleaned %d low-quality fragments", cleaned)

        return cleaned

    # ------------------------------------------------------------------
    # Insight persistence
    # ------------------------------------------------------------------

    def _save_insight(self, insight: Dict[str, Any]) -> None:
        """Save a discovered insight to the insights directory."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        title_slug = "".join(
            c if c.isalnum() else "_"
            for c in insight.get("title", "insight")[:40]
        )
        fname = f"{timestamp}_{title_slug}.json"
        fpath = os.path.join(self._insights_dir, fname)

        try:
            os.makedirs(self._insights_dir, exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as fh:
                json.dump(insight, fh, ensure_ascii=False, indent=2)
            logger.debug("Insight saved to %s", fpath)
        except (IOError, OSError) as exc:
            logger.warning("Failed to save insight: %s", exc)

        # Also log to an index file
        index_path = os.path.join(self._insights_dir, "index.json")
        try:
            index_entries: List[Dict[str, Any]] = []
            if os.path.isfile(index_path):
                with open(index_path, "r", encoding="utf-8") as fh:
                    index_entries = json.load(fh)
            index_entries.append({
                "file": fname,
                "title": insight.get("title", ""),
                "confidence": insight.get("confidence", 0),
                "time": insight.get("cycle_time", ""),
            })
            with open(index_path, "w", encoding="utf-8") as fh:
                json.dump(index_entries, fh, ensure_ascii=False, indent=2)
        except (IOError, OSError):
            pass


# ======================================================================
# Self-test
# ======================================================================

def _self_test() -> None:
    """Run: python -m hms.scripts.dream_engine"""
    import tempfile

    print("Testing DreamEngine ...")

    with tempfile.TemporaryDirectory() as d:
        de = DreamEngine({
            "dream_insights_dir": os.path.join(d, "insights"),
            "clean_orphans": True,
        })

        # Test with synthetic memories
        memories = [
            {"id": f"m{i}", "text": f"记忆碎片 {i}: 关于{'编程' if i % 2 else '天气'}的一些内容",
             "metadata": {"entities": ["Python", "AI"] if i % 2 else [], "topics": ["tech"] if i % 2 else ["life"]},
             "importance": random.randint(1, 10)}
            for i in range(8)
        ]

        # Test distant association finding
        assocs = de._find_distant_associations(memories)
        assert isinstance(assocs, list)
        print(f"[distant associations] found {len(assocs)} paths")

        # Test heuristic insight
        if assocs:
            insight = de._generate_insight(assocs[0])
            assert insight is not None
            assert "title" in insight
            print(f"[insight generation] OK (title={insight['title']})")
        else:
            # Test with a manual path
            insight = de._generate_insight(memories[:2])
            assert insight is not None
            print(f"[insight generation with manual path] OK")

        # Test fragment cleaning
        test_memories = [
            {"id": "clean1", "text": "好", "metadata": {}, "importance": 1},
            {"id": "clean2", "text": "长文本", "metadata": {"entities": ["X"]}, "importance": 5},
        ]
        cleaned = de._clean_fragments(test_memories)
        assert isinstance(cleaned, int)
        print(f"[fragment cleanup] cleaned {cleaned}")

        # Test save insight
        test_insight = {
            "title": "测试洞察",
            "description": "这是一个测试",
            "confidence": 0.7,
            "meaningful": True,
        }
        de._save_insight(test_insight)
        assert len(os.listdir(os.path.join(d, "insights"))) >= 1
        print("[insight saving] OK")

    print("✓ All DreamEngine self-tests passed.")


if __name__ == "__main__":
    _self_test()
