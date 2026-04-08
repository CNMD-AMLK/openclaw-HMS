"""
HMS v5 — Collision Engine (adapted from hms-v4 scripts/collision.py).
"""

from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List, Optional

from hms.utils.llm import LLMAnalyzer
from hms.utils.text import tokenize

logger = logging.getLogger("hms.collision")


class CollisionEngine:
    def __init__(self, config: Dict[str, Any], *, llm: Optional[LLMAnalyzer] = None) -> None:
        self.cfg = config
        self.llm = llm or LLMAnalyzer(config)
        self.embed_cache = None
        self._embed_threshold = config.get("embed_similarity_threshold", 0.3)
        self._embed_max_candidates = config.get("embed_max_candidates", 10)
        self._llm_collision_threshold = config.get("llm_collision_threshold", 0.85)
        self._heuristic_min_threshold = config.get("heuristic_min_threshold", 0.6)

    def set_embed_cache(self, cache: Any) -> None:
        self.embed_cache = cache

    def collide(self, new_perception: Dict[str, Any],
                retrieved_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        candidates = retrieved_memories

        if not candidates:
            return {
                "contradictions": [], "reinforcements": [], "associations": [],
                "new_insights": [], "method": "no_candidates",
            }

        new_text = new_perception.get("text_for_store", "")
        if self.embed_cache and new_text:
            try:
                similar = self.embed_cache.find_similar(
                    query=new_text, candidates=retrieved_memories,
                    top_k=self._embed_max_candidates, threshold=self._embed_threshold,
                )
                candidates = [c for c, _ in similar] if similar else []
            except Exception:
                pass

        if not candidates:
            return {
                "contradictions": [], "reinforcements": [], "associations": [],
                "new_insights": [], "method": "embed_filtered_empty",
            }

        if self.embed_cache and new_text:
            max_sim = max(
                (self.embed_cache.similarity(new_text, c.get("text", "")) for c in candidates),
                default=0.0,
            )
            if max_sim < self._heuristic_min_threshold:
                return {
                    "contradictions": [], "reinforcements": [], "associations": [],
                    "new_insights": [], "method": "embed_skip", "max_similarity": round(max_sim, 3),
                }
            if max_sim < self._llm_collision_threshold:
                result = self._heuristic_collision(new_perception, candidates)
                result["method"] = "heuristic_by_similarity"
                result["max_similarity"] = round(max_sim, 3)
                return result

        llm_result = self._llm_collide(new_perception, candidates)
        if llm_result:
            llm_result["method"] = "llm"
            return llm_result

        result = self._heuristic_collision(new_perception, candidates)
        result["method"] = "heuristic_fallback"
        return result

    def _llm_collide(self, new_perception: Dict, candidates: List[Dict]) -> Optional[Dict]:
        """Simple LLM collision via structured prompt."""
        mem_lines = [f"[{i}] {m.get('text', '')[:150]}" for i, m in enumerate(candidates)]
        prompt = f"""分析新感知与已有记忆是否存在矛盾、强化或关联。

新感知: {new_perception.get('text_for_store', '')[:300]}

已有记忆:
{chr(10).join(mem_lines)}

请以JSON格式返回：
{{
  "contradictions": [{{"existing_id": "id", "severity": "low|medium|high", "reason": "原因"}}],
  "reinforcements": [{{"existing_id": "id", "confidence_delta": 0.05, "reason": "原因"}}],
  "associations": [{{"existing_id": "id", "relation_type": "类型", "confidence": 0.5}}],
  "new_insights": [{{"content": "推断内容", "confidence": 0.6, "based_on": ["id"]}}]
}}"""
        raw = self.llm._call_llm(prompt, max_tokens=1000, temperature=0.1)
        if not raw:
            return None
        return self.llm._parse_json_response(raw)

    def _heuristic_collision(self, new_perception: Dict, retrieved: List[Dict]) -> Dict[str, Any]:
        """Heuristic collision detection."""
        contradictions, reinforcements, associations = [], [], []
        new_text = (new_perception.get("text_for_store", "") or "").lower()
        new_entities = {
            (e.get("name") if isinstance(e, dict) else str(e)).lower()
            for e in new_perception.get("entities", [])
        }

        negation_words = {"不", "没", "无", "非", "未", "否", "别", "不", "not", "no", "never", "n't"}
        contradiction_patterns = [re.compile(p) for p in ["但是", "然而", "却", "相反", "but", "however"]]

        def has_negation(text: str) -> bool:
            return any(w in text for w in negation_words) or any(p.search(text) for p in contradiction_patterns)

        for mem in retrieved:
            mem_id = mem.get("id", "")
            mem_text = (mem.get("text", "") or "").lower()
            mem_entities = set()
            meta = mem.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            for e in meta.get("entities", []):
                mem_entities.add(str(e).lower())

            shared = new_entities & mem_entities
            if not shared:
                continue

            new_neg = has_negation(new_text)
            mem_neg = has_negation(mem_text)

            if new_neg != mem_neg:
                contradictions.append({
                    "existing_id": mem_id,
                    "severity": "medium",
                    "reason": "negation_mismatch",
                })
            else:
                reinforcements.append({
                    "existing_id": mem_id,
                    "confidence_delta": 0.05,
                    "reason": f"shared entities: {', '.join(list(shared)[:2])}",
                })

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
        }

    def execute_results(self, collision_result: Dict, memory_store_func=None,
                        new_memory_id: str = "") -> Dict[str, Any]:
        report = {"stored_inferences": 0, "graph_edges": 0, "errors": []}
        for inf in collision_result.get("new_insights", []):
            if memory_store_func and inf.get("confidence", 0) >= 0.6:
                try:
                    memory_store_func(
                        text=f"[推断] {inf.get('content', '')}",
                        category="fact", importance=5,
                        metadata=json.dumps({"source": "inferred", "derived_from": inf.get("based_on", [])}, ensure_ascii=False),
                    )
                    report["stored_inferences"] += 1
                except Exception as e:
                    report["errors"].append(str(e))
        return report
