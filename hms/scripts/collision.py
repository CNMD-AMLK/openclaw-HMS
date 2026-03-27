"""
HMS v2 — Collision Engine.

LLM-driven semantic collision detection.
Replaces v1's keyword-overlap approach with deep semantic analysis.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .llm_analyzer import LLMAnalyzer


class CollisionEngine:
    """
    Detects contradictions, reinforcements, associations, and inferences
    between new perceptions and existing memories using LLM analysis.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self.llm = LLMAnalyzer(self.cfg)

    def collide(
        self,
        new_perception: Dict[str, Any],
        retrieved_memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run full collision pipeline.

        Tries LLM first, falls back to heuristic if unavailable.
        """
        # Try LLM collision
        llm_result = self.llm.collide(new_perception, retrieved_memories)
        if llm_result:
            llm_result["method"] = "llm"
            return llm_result

        # Fallback: simple heuristic collision
        return self._heuristic_collision(new_perception, retrieved_memories)

    def _heuristic_collision(
        self,
        new_perception: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Lightweight heuristic collision when LLM is unavailable."""
        contradictions = []
        reinforcements = []
        associations = []

        new_text = new_perception.get("text_for_store", "").lower()
        new_entities = set()
        for e in new_perception.get("entities", []):
            if isinstance(e, dict):
                new_entities.add(e.get("name", "").lower())
            else:
                new_entities.add(str(e).lower())

        for mem in retrieved:
            mem_id = mem.get("id", "")
            mem_text = (mem.get("text", "") or "").lower()
            mem_meta = mem.get("metadata") or {}

            # Extract entities from metadata
            mem_entities = set()
            for e in mem_meta.get("entities", []):
                mem_entities.add(str(e).lower())

            shared = new_entities & mem_entities
            if not shared:
                continue

            # Simple negation detection
            new_neg = any(w in new_text for w in ["不", "没", "无", "非"])
            mem_neg = any(w in mem_text for w in ["不", "没", "无", "非"])

            if new_neg != mem_neg:
                # Potential contradiction
                contradictions.append({
                    "existing_id": mem_id,
                    "severity": "medium",
                    "reason": "negation_mismatch",
                    "suggestion": "review",
                })
            else:
                # Potential reinforcement
                reinforcements.append({
                    "existing_id": mem_id,
                    "confidence_delta": 0.05,
                    "reason": f"shared entities: {', '.join(list(shared)[:2])}",
                })

            # Association
            associations.append({
                "existing_id": mem_id,
                "relation_type": "co_mentioned",
                "confidence": 0.5,
                "reason": f"shared: {', '.join(list(shared)[:3])}",
            })

        return {
            "contradictions": contradictions,
            "reinforcements": reinforcements,
            "associations": associations,
            "new_insights": [],
            "method": "heuristic",
        }

    def execute_results(
        self,
        collision_result: Dict[str, Any],
        memory_store_func: Optional[Callable] = None,
        gm_record_func: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute collision results: store inferences, record graph edges.
        """
        report = {"stored_inferences": 0, "graph_edges": 0, "errors": []}

        # Store inferences as new memories
        for inf in collision_result.get("new_insights", []):
            if memory_store_func and inf.get("confidence", 0) >= 0.6:
                try:
                    content = inf.get("content", "")
                    based_on = inf.get("based_on", [])
                    metadata = json.dumps({
                        "source": "inferred",
                        "belief_confidence": inf.get("confidence", 0.5),
                        "derived_from": based_on,
                        "memory_type": "semantic",
                    }, ensure_ascii=False)
                    memory_store_func(
                        text=f"[推断] {content}",
                        category="fact",
                        importance=5,
                        metadata=metadata,
                    )
                    report["stored_inferences"] += 1
                except Exception as e:
                    report["errors"].append(f"inference_store: {e}")

        # Record associations as graph edges
        for assoc in collision_result.get("associations", []):
            if gm_record_func and assoc.get("confidence", 0) >= 0.5:
                try:
                    gm_record_func(
                        source=assoc.get("existing_id", ""),
                        target="new_perception",
                        relation=assoc.get("relation_type", "related"),
                        context=assoc.get("reason", ""),
                    )
                    report["graph_edges"] += 1
                except Exception:
                    pass

        return report


# ======================================================================
# Self-test
# ======================================================================


def _self_test():
    """Run: python -m hms.scripts.collision"""
    engine = CollisionEngine()

    perception = {
        "text_for_store": "用户说 Python 比 Java 更好用",
        "entities": [{"name": "Python", "type": "tool"}, {"name": "Java", "type": "tool"}],
    }
    memories = [
        {
            "id": "mem_001",
            "text": "用户之前说 Java 是最好的语言",
            "metadata": {"entities": ["Java"]},
        },
    ]

    result = engine.collide(perception, memories)
    assert "contradictions" in result
    assert "reinforcements" in result
    print(f"[collision] contradictions={len(result['contradictions'])} method={result.get('method')}")

    print("✓ All self-tests passed.")


if __name__ == "__main__":
    _self_test()
