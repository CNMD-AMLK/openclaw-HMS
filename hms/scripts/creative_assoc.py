"""
HMS v4 — Creative Association Engine.

Finds non-obvious connections between disparate topics by:
  1. Multi-hop graph traversal from topic A to topic B
  2. LLM evaluation of whether the connection is genuinely creative
  3. Generating insight reports with the discoveries

This is inspired by lateral thinking and the 'bisociation' theory of creativity.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .config_loader import Config

logger = logging.getLogger("hms.creative_assoc")


class CreativeAssociator:
    """
    Creative topic association engine.

    Discovers unexpected but meaningful links between topics that would
    normally be considered unrelated.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self._max_hops = self.cfg.get("creative_max_hops", 3)
        self._max_topics = self.cfg.get("creative_max_topics", 20)
        self._min_creativity = self.cfg.get("creative_min_score", 0.6)
        logger.debug(
            "CreativeAssociator initialized (max_hops=%d, min_creativity=%.2f)",
            self._max_hops, self._min_creativity,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def creative_link(
        self,
        topic_a: str,
        topic_b: str,
        max_hops: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Find a creative link between two topics.

        Args:
            topic_a: first topic/entity
            topic_b: second topic/entity
            max_hops: max number of intermediate hops (override default)

        Returns:
            Dict with the link path, LLM evaluation, and creativity score.
        """
        hops = max_hops or self._max_hops

        # Build hop paths
        paths = self._graph_hop(topic_a, max_hops=hops)

        if not paths:
            return {
                "topic_a": topic_a,
                "topic_b": topic_b,
                "linked": False,
                "reason": "no hop path found",
            }

        # Find paths that lead to topic_b
        relevant_paths = []
        for path in paths:
            tail = path[-1].get("entity", "").lower()
            b_lower = topic_b.lower()
            if tail == b_lower or b_lower in tail or tail in b_lower:
                relevant_paths.append(path)

        if not relevant_paths:
            # Try to find any creative connection
            # Pick the most interesting path
            for path in paths:
                evaluation = self._evaluate_link(path)
                if evaluation.get("creative", False):
                    return {
                        "topic_a": topic_a,
                        "topic_b": topic_b,
                        "linked": True,
                        "path": self._format_path(path),
                        "evaluation": evaluation,
                        "creativity_score": evaluation.get("creativity", 0),
                    }

            return {
                "topic_a": topic_a,
                "topic_b": topic_b,
                "linked": False,
                "reason": "no creative connection found",
            }

        # Evaluate the best path
        best_path = max(relevant_paths, key=lambda p: len(p))
        evaluation = self._evaluate_link(best_path)

        return {
            "topic_a": topic_a,
            "topic_b": topic_b,
            "linked": True,
            "path": self._format_path(best_path),
            "evaluation": evaluation,
            "creativity_score": evaluation.get("creativity", 0),
        }

    # ------------------------------------------------------------------
    # Graph multi-hop traversal
    # ------------------------------------------------------------------

    def _graph_hop(
        self,
        start: str,
        max_hops: int = 3,
    ) -> List[List[Dict[str, Any]]]:
        """
        Multi-hop traversal starting from the given entity/topic.

        Uses the knowledge graph to find entities reachable within max_hops.
        Tracks seen endpoint pairs to avoid duplicate paths and cycles.
        """
        # Build entity-topic graph from memories
        graph: Dict[str, Set[str]] = {}  # entity → {related entities}
        memories = self._load_memories()

        for mem in memories:
            meta = mem.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, ValueError):
                    meta = {}
            if not isinstance(meta, dict):
                meta = {}

            entities: List[str] = []
            for ent in meta.get("entities", []):
                ent_str = ent.lower().strip() if isinstance(ent, str) else ent.get("name", "").lower().strip()
                if ent_str:
                    entities.append(ent_str)
            topics: List[str] = [t.lower().strip() for t in meta.get("topics", []) if t]

            # All entities in this memory are connected to each other
            all_nodes = entities + topics
            for i, node_a in enumerate(all_nodes):
                for node_b in all_nodes[i + 1:]:
                    graph.setdefault(node_a, set()).add(node_b)
                    graph.setdefault(node_b, set()).add(node_a)

        # BFS from start
        start_lower = start.lower().strip()
        if start_lower not in graph:
            return []  # Entity not in graph

        from collections import deque
        paths: List[List[Dict[str, Any]]] = []
        seen_pairs: Set[tuple] = set()  # track (start, end) pairs to avoid duplicates
        queue: deque = deque([(start_lower, [{
            "entity": start_lower,
            "role": "source",
            "memories": self._find_memories_with(start_lower, memories),
        }])])
        visited: Set[str] = {start_lower}
        found_at_depth: Dict[int, int] = {}  # depth → count found

        while queue:
            current_node, current_path = queue.popleft()
            depth = len(current_path) - 1

            if depth >= max_hops:
                # Record this path at max depth
                end_entity = current_path[-1]["entity"]
                pair = (start_lower, end_entity)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    paths.append(list(current_path))
                continue

            # BFS expansion
            neighbors = graph.get(current_node, set())
            if not neighbors:
                end_entity = current_path[-1]["entity"]
                pair = (start_lower, end_entity)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    paths.append(list(current_path))
                continue

            for neighbor in neighbors:
                if neighbor not in visited or depth < 2:
                    new_path = current_path + [{
                        "entity": neighbor,
                        "role": "intermediate" if depth < max_hops - 1 else "target",
                        "memories": self._find_memories_with(neighbor, memories),
                    }]

                    if depth < max_hops - 1:
                        visited.add(neighbor)
                        queue.append((neighbor, new_path))
                        if depth == 0:
                            found_at_depth[depth] = found_at_depth.get(depth, 0) + 1

                    # Also record this intermediate path (dedup by endpoint pair)
                    end_entity = neighbor
                    pair = (start_lower, end_entity)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        paths.append(list(new_path))

        # Limit to prevent combinatorial explosion
        paths.sort(key=lambda p: len(p), reverse=True)
        return paths[:50]  # cap at 50 paths

    @staticmethod
    def _find_memories_with(token: str, memories: List[Dict]) -> List[str]:
        """Find memory IDs that contain the given token."""
        return [
            m.get("id", "") for m in memories
            if token in m.get("text", "").lower() or
            token in " ".join(m.get("metadata", {}).get("entities", [])).lower() or
            token in " ".join(m.get("metadata", {}).get("topics", [])).lower()
        ]

    # ------------------------------------------------------------------
    # Link evaluation via LLM
    # ------------------------------------------------------------------

    def _evaluate_link(self, path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM to evaluate whether a multi-hop path represents a
        genuinely creative and meaningful connection.
        """
        if len(path) < 2:
            return {"creative": False, "reason": "path too short"}

        prompt_parts = ["分析以下实体/概念之间的关联路径，判断是否存在创造性的关联：\n\n"]
        for i, step in enumerate(path, 1):
            ent = step.get("entity", "")
            role = step.get("role", "")
            mem_ids = step.get("memories", [])
            prompt_parts.append(f"步骤 {i} [{role}]: {ent}")
            if mem_ids:
                prompt_parts.append(f"  相关记忆: {', '.join(mem_ids[:3])}")

        prompt_parts.extend([
            "",
            "请判断：",
            "1. 这条路径是否揭示了有趣的、非显而易见的关联？",
            "2. 这种关联是否在现实中有意义？",
            "3. 能否基于此产生创新性的洞察？",
            "",
            "请以JSON格式回答：",
            '{"creative": true/false, "creativity": 0.0-1.0, '
            '"explanation": "说明为什么有趣", '
            '"potential_insight": "可能的创新洞察", '
            '"risk": "误报风险 (low/medium/high)"}',
        ])

        prompt = "\n".join(prompt_parts)

        try:
            from hms.scripts.llm_analyzer import LLMAnalyzer
            analyzer = LLMAnalyzer()
            raw = analyzer.analyze(prompt, max_tokens=600, temperature=0.6)
            result = analyzer._parse_json_response(raw) if isinstance(raw, str) else None

            if isinstance(result, dict):
                result.setdefault("creative", False)
                result.setdefault("creativity", 0.0)
                return result
        except Exception as exc:
            logger.debug("LLM link evaluation failed: %s", exc)

        return self._heuristic_evaluate(path)

    @staticmethod
    def _heuristic_evaluate(path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Heuristic evaluation when LLM is unavailable."""
        path_len = len(path)
        entities = [step.get("entity", "") for step in path]

        # Long paths through diverse domains are more likely creative
        unique_entities = len(set(entities))
        diversity_score = unique_entities / max(path_len, 1)

        # Base creativity: moderate
        creativity = min(0.7, 0.3 + diversity_score * 0.3 + (0.1 if path_len >= 3 else 0))

        return {
            "creative": creativity >= 0.4,
            "creativity": round(creativity, 2),
            "explanation": f"通过 {path_len} 个步骤连接 {unique_entities} 个不同实体",
            "potential_insight": f"{' → '.join(entities)} 存在间接关联",
            "risk": "medium" if path_len >= 3 else "low",
            "heuristic": True,
        }

    # ------------------------------------------------------------------
    # Format path
    # ------------------------------------------------------------------

    @staticmethod
    def _format_path(path: List[Dict[str, Any]]) -> List[str]:
        """Format a path for human-readable output."""
        return [f"{i+1}. {step['entity']} [{step['role']}]"
                for i, step in enumerate(path)]

    # ------------------------------------------------------------------
    # Generate insights report
    # ------------------------------------------------------------------

    def generate_insights(self) -> Dict[str, Any]:
        """
        Auto-generate creative insight report.

        Scans all known topics/entities, finds creative connections,
        and produces a summary report.
        """
        insights: List[Dict[str, Any]] = []
        errors: List[str] = []

        # Gather all known topics
        all_topics: List[str] = []
        all_entities: List[str] = []
        seen: set = set()

        memories = self._load_memories()
        for mem in memories:
            meta = mem.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, ValueError):
                    meta = {}
            if not isinstance(meta, dict):
                meta = {}

            for topic in meta.get("topics", []):
                if topic.lower() not in seen:
                    seen.add(topic.lower())
                    all_topics.append(topic)

            for ent in meta.get("entities", []):
                ent_str = ent if isinstance(ent, str) else ent.get("name", "")
                if ent_str.lower() not in seen:
                    seen.add(ent_str.lower())
                    all_entities.append(ent_str)

        # Try creative links between topics and entities
        candidates = all_topics[:self._max_topics] + all_entities[:self._max_topics]

        # Limit comparisons: O(n^2) can explode
        import random
        if len(candidates) > 15:
            # Pick random pairs
            pairs = random.sample(
                [(a, b) for i, a in enumerate(candidates) for b in candidates[i + 1:]],
                min(30, len(candidates) * 2),
            )
        else:
            pairs = [
                (a, b) for i, a in enumerate(candidates) for b in candidates[i + 1:]
            ]

        for topic_a, topic_b in pairs:
            try:
                result = self.creative_link(topic_a, topic_b)
                if result.get("linked") and result.get("creativity_score", 0) >= self._min_creativity:
                    insights.append({
                        "topic_a": topic_a,
                        "topic_b": topic_b,
                        "path": result.get("path", []),
                        "creativity_score": result.get("creativity_score", 0),
                        "evaluation": result.get("evaluation", {}),
                    })
            except Exception as exc:
                errors.append(f"{topic_a} ↔ {topic_b}: {exc}")

        # Sort by creativity score
        insights.sort(key=lambda i: i.get("creativity_score", 0), reverse=True)

        report = {
            "time": datetime.now(timezone.utc).isoformat(),
            "topics_scanned": len(all_topics),
            "entities_scanned": len(all_entities),
            "pairs_evaluated": len(pairs),
            "insights_count": len(insights),
            "top_insights": insights[:10],
            "errors": errors[:5],
        }

        # Save report
        try:
            cache_dir = Config.get().get("cache_dir", "cache")
            report_dir = os.path.join(cache_dir, "creative_reports")
            os.makedirs(report_dir, exist_ok=True)
            fname = f"creative_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            with open(os.path.join(report_dir, fname), "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)
        except (IOError, OSError) as exc:
            logger.debug("Failed to save creative report: %s", exc)

        return report

    # ------------------------------------------------------------------
    # Memory loading (shared)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_memories() -> List[Dict[str, Any]]:
        """Load all available memories."""
        memories: List[Dict[str, Any]] = []
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
            logger.debug("Creative assoc memory loading failed: %s", exc)
        return memories


# ======================================================================
# Self-test
# ======================================================================

def _self_test() -> None:
    """Run: python -m hms.scripts.creative_assoc"""
    print("Testing CreativeAssociator ...")

    ca = CreativeAssociator({})

    # Test heuristic evaluation
    heuristic_path = [
        {"entity": "python", "role": "source"},
        {"entity": "programming", "role": "intermediate"},
        {"entity": "automation", "role": "target"},
    ]
    result = ca._heuristic_evaluate(heuristic_path)
    assert "creative" in result
    assert "creativity" in result
    print(f"[heuristic eval] OK (creative={result['creative']}, score={result['creativity']})")

    # Test path formatting
    formatted = ca._format_path(heuristic_path)
    assert len(formatted) == 3
    print(f"[path format] OK: {formatted}")

    # Test generate_insights (no LLM available, heuristic path)
    report = ca.generate_insights()
    assert "insights_count" in report
    assert "top_insights" in report
    print(f"[generate insights] OK (insights={report['insights_count']})")

    print("✓ All CreativeAssociator self-tests passed.")


if __name__ == "__main__":
    _self_test()
