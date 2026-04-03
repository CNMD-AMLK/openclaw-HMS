"""
HMS v4 — Reconstructive Recall Engine.

Unlike exact retrieval, reconstructive recall mimics how human memory works:
instead of fetching an exact stored record, it retrieves memory fragments
and reconstructs a coherent answer weighted by the current context.

Key methods:
    recall(query, context)  → main entry point
    _search_fragments(query, top_k)  → find relevant memory fragments
    _weight_by_context(fragments, context)  → weight by current context
    _llm_synthesize(fragments, query)  → LLM reconstructive synthesis
    _tag_confidence(result)  → tag as exact / reconstructed
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config_loader import Config

logger = logging.getLogger("hms.reconstructive_recall")


# ---------------------------------------------------------------------------
# Internal cache to avoid re-synthesizing the same query within a short window.
# ---------------------------------------------------------------------------
class _RRCache:
    """Short-lived LRU cache for recall results."""

    def __init__(self, max_size: int = 64, ttl_secs: int = 300) -> None:
        self._max = max_size
        self._ttl = ttl_secs
        self._store: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry["_ts"] > self._ttl:
            del self._store[key]
            return None
        return entry

    def put(self, key: str, value: Dict[str, Any]) -> None:
        # Evict oldest on overflow
        if len(self._store) >= self._max:
            oldest = min(self._store, key=lambda k: self._store[k]["_ts"])
            del self._store[oldest]
        value["_ts"] = time.monotonic()
        self._store[key] = value

    def clear(self) -> None:
        self._store.clear()


_rr_cache = _RRCache()


def _cache_key(query: str, context_seed: str) -> str:
    """Deterministic cache key."""
    import hashlib
    raw = f"{query}|{context_seed}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ======================================================================
# ReconstructiveRecaller
# ======================================================================

class ReconstructiveRecaller:
    """
    Reconstructive memory recall.

    Instead of 'search → return exact matches', this engine:
      1. Retrieves memory fragments that are loosely related
      2. Weights them by the current conversational context
      3. Synthesises a reconstructed answer via LLM
      4. Tags the result with a confidence label (extracted / reconstructed)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self._top_k = self.cfg.get("rr_top_k", 10)
        self._synthesis_threshold = self.cfg.get("rr_synthesis_threshold", 0.4)
        self._context_weight = self.cfg.get("rr_context_weight", 0.3)
        logger.debug(
            "ReconstructiveRecaller initialized (top_k=%d, threshold=%.2f)",
            self._top_k, self._synthesis_threshold,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Reconstructive recall.

        Args:
            query: the user's question / topic to recall
            context: current conversational context (optional)

        Returns:
            Dict with 'answer' (reconstructed text), 'fragments' (used),
            'confidence' (0-1), 'label' (extracted | reconstructed).
        """
        if not query.strip():
            return self._empty_result(query)

        # Check cache
        ctx_seed = json.dumps(
            context.get("perception", {}) if context else {},
            sort_keys=True, ensure_ascii=False,
        )
        key = _cache_key(query, ctx_seed)
        cached = _rr_cache.get(key)
        if cached is not None:
            cached["_cached"] = True
            return cached

        # 1. Search fragments
        fragments = self._search_fragments(query, top_k=self._top_k)

        if not fragments:
            return self._empty_result(query)

        # 2. Weight by context
        if context:
            fragments = self._weight_by_context(fragments, context)

        # 3. Decide: exact match available? or need synthesis?
        best_score = max((f.get("score", 0.0) for f in fragments), default=0.0)

        if best_score >= self._synthesis_threshold and len(fragments) == 1:
            # Exact-enough match — return directly
            result = {
                "answer": fragments[0].get("text", ""),
                "memory_id": fragments[0].get("id", ""),
                "fragments": fragments,
                "confidence": round(best_score, 3),
                "label": "extracted",
                "query": query,
                "method": "exact_retrieval",
            }
        else:
            # Need LLM synthesis
            synthesis = self._llm_synthesize(fragments, query)
            tagged = self._tag_confidence(synthesis)
            tagged["query"] = query
            tagged["method"] = "reconstructed"
            result = tagged
            # Ensure fragments are included
            result["fragments_used"] = [
                {"id": f.get("id", ""), "text": f.get("text", "")[:200],
                 "score": f.get("score", 0.0)}
                for f in fragments[:5]
            ]

        _rr_cache.put(key, result)
        return result

    # ------------------------------------------------------------------
    # Fragment search
    # ------------------------------------------------------------------

    def _search_fragments(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memory fragments.

        Uses embedding similarity + keyword overlap + temporal recency.
        Falls back to local text search if no memory adapter is available.
        """
        try:
            from hms.scripts.memory_manager import MemoryAdapter
            from hms.scripts.config_loader import Config as CfgLoader
            cfg = CfgLoader.get()
            adapter = MemoryAdapter(cfg)
            try:
                memories = adapter.recall(query=query, top_k=top_k)
            finally:
                adapter.close()

            return [
                {
                    "id": m.get("id", ""),
                    "text": m.get("text", ""),
                    "metadata": m.get("metadata", {}),
                    "importance": m.get("importance", 5),
                    "score": 0.5,  # baseline — will be re-weighted
                    "created_at": m.get("created_at", ""),
                }
                for m in (memories or [])
            ]
        except Exception as exc:
            logger.debug("Fragment search via adapter failed: %s", exc)
            # Fallback: keyword-based local search
            return self._local_keyword_search(query, top_k)

    @staticmethod
    def _local_keyword_search(query: str, top_k: int) -> List[Dict[str, Any]]:
        """Lightweight keyword search as fallback."""
        # Try to read from local cache files
        results: List[Dict[str, Any]] = []
        from hms.scripts.utils import tokenize
        query_tokens = set(tokenize(query.lower()))
        if not query_tokens:
            return results

        cache_dir = Config.get().get("cache_dir", "cache")
        for filename in ["beliefs.json", "cognitive_fingerprint.json"]:
            fpath = os.path.join(cache_dir, filename)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    entries = data
                elif isinstance(data, dict):
                    entries = list(data.values())
                else:
                    continue
                for entry in entries:
                    text = entry.get("text", "") or entry.get("content", "")
                    if not text:
                        continue
                    entry_tokens = set(tokenize(text.lower()))
                    overlap = query_tokens & entry_tokens
                    score = len(overlap) / max(len(query_tokens), 1)
                    if score > 0.1:
                        results.append({
                            "id": entry.get("id", ""),
                            "text": text,
                            "metadata": entry.get("metadata", {}),
                            "importance": entry.get("importance", 3),
                            "score": round(score, 3),
                            "created_at": entry.get("created_at", ""),
                        })
            except (json.JSONDecodeError, IOError, OSError):
                continue

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Context weighting
    # ------------------------------------------------------------------

    def _weight_by_context(
        self,
        fragments: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Re-weight fragments based on current conversational context.

        Scoring factors:
          - Importance of the memory
          - Recency
          - Topical overlap with current perception
          - Emotional alignment
        """
        from hms.scripts.utils import tokenize

        perception = context.get("perception", {})
        current_topics: set = set()
        if isinstance(perception, dict):
            current_topics = set(tokenize(" ".join(perception.get("topics", []))))
            current_entities: set = set()
            for ent in perception.get("entities", []):
                if isinstance(ent, dict):
                    current_entities.add(ent.get("name", "").lower())
                else:
                    current_entities.add(str(ent).lower())

        for frag in fragments:
            score = frag.get("score", 0.5)

            # Importance boost (0-0.2)
            importance = frag.get("importance", 5)
            score += (importance / 10.0) * 0.2

            # Recency boost (0-0.15)
            created = frag.get("created_at", "")
            if created:
                try:
                    created_dt = datetime.fromisoformat(created)
                    if created_dt.tzinfo is None:
                        created_dt = created_dt.replace(tzinfo=timezone.utc)
                    hours_ago = max(0, (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600)
                    recency = max(0, 1 - hours_ago / (24 * 30))  # 30-day decay
                    score += recency * 0.15
                except (ValueError, TypeError):
                    pass

            # Topical overlap boost (0-0.2)
            frag_meta = frag.get("metadata", {})
            if isinstance(frag_meta, str):
                try:
                    frag_meta = json.loads(frag_meta)
                except (json.JSONDecodeError, ValueError):
                    frag_meta = {}

            frag_topics: set = set()
            if isinstance(frag_meta, dict):
                frag_topics = set(tokenize(" ".join(frag_meta.get("topics", []))))
                frag_entities = set(
                    e.lower() for e in frag_meta.get("entities", [])
                )

            if current_topics and frag_topics:
                topic_overlap = len(current_topics & frag_topics) / max(len(current_topics), 1)
                score += topic_overlap * 0.2

            if current_entities and frag_entities:
                entity_match = len(current_entities & frag_entities) > 0
                if entity_match:
                    score += 0.1

            frag["score"] = round(min(1.0, score), 3)

        fragments.sort(key=lambda f: f["score"], reverse=True)
        return fragments

    # ------------------------------------------------------------------
    # LLM synthesis
    # ------------------------------------------------------------------

    def _llm_synthesize(
        self,
        fragments: List[Dict[str, Any]],
        query: str,
    ) -> Dict[str, Any]:
        """
        Synthesize a reconstructed answer using LLM.

        Constructs a prompt with the retrieved fragments and asks the LLM
        to reconstruct an answer, clearly marking what it's certain about
        vs what it's inferring.
        """
        # Build the synthesis prompt
        fragment_texts = []
        for i, frag in enumerate(fragments[:7][:7], 1):
            text = frag.get("text", "").replace("\n", " ")[:300]
            if text:
                fragment_texts.append(f"[{i}] {text} (score={frag.get('score', 0):.2f})")

        if not fragment_texts:
            return self._empty_result(query)

        prompt = (
            "你是一位记忆重建专家。根据以下记忆碎片，重建对用户问题的回答。\n"
            "请注意：\n"
            "1. 只基于提供的记忆碎片回答，不要编造事实\n"
            "2. 如果某些信息不够确定，请明确标注\n"
            "3. 将记忆碎片中的信息进行综合，形成连贯的回答\n\n"
            f"用户问题：{query}\n\n"
            "记忆碎片：\n" +
            "\n".join(fragment_texts) +
            "\n\n"
            "请以JSON格式回答：\n"
            '{"answer": "重建后的回答", "confidence": 0.0-1.0, '
            '"note": "确定性说明（如哪些是确切的，哪些是推断的）"}'
        )

        try:
            from hms.scripts.llm_analyzer import LLMAnalyzer
            analyzer = LLMAnalyzer()
            raw = analyzer.analyze(prompt, max_tokens=1000, temperature=0.3)
            result = analyzer._parse_json_response(raw) if isinstance(raw, str) else None

            if result and isinstance(result, dict) and "answer" in result:
                return {
                    "answer": result.get("answer", ""),
                    "confidence": min(1.0, max(0.0, float(result.get("confidence", 0.5)))),
                    "note": result.get("note", ""),
                    "fragments_count": len(fragments),
                }
            else:
                # Parse raw text if JSON parsing failed
                return {
                    "answer": raw if isinstance(raw, str) else str(result),
                    "confidence": 0.4,
                    "note": "LLM response was not valid JSON; raw output used",
                    "fragments_count": len(fragments),
                    "parse_error": True,
                }
        except Exception as exc:
            logger.debug("LLM synthesis failed: %s", exc)
            # Fallback: concatenate fragments weighted by score
            sorted_frags = sorted(fragments, key=lambda f: f.get("score", 0), reverse=True)
            concatenated = "；".join(f.get("text", "") for f in sorted_frags[:3] if f.get("text"))
            return {
                "answer": concatenated or "没有找到相关记忆。",
                "confidence": max((f.get("score", 0.0) for f in sorted_frags[:1]), default=0.0),
                "note": "LLM synthesizer unavailable; using concatenated fragments",
                "fragments_count": len(fragments),
                "fallback": True,
            }

    # ------------------------------------------------------------------
    # Confidence tagging
    # ------------------------------------------------------------------

    @staticmethod
    def _tag_confidence(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tag the recall result with a human-readable confidence label.

        Labels:
          - "exact"      : score >= 0.85, single high-confidence match
          - "reconstructed": score 0.5-0.85, LLM-synthesized
          - "inferred"   : score < 0.5, mostly speculative
        """
        confidence = result.get("confidence", 0.0)
        fragments_count = result.get("fragments_count", 0)

        if confidence >= 0.85:
            label = "exact"
        elif confidence >= 0.5:
            label = "reconstructed"
        else:
            label = "inferred"

        result["label"] = label
        result["fragments_used"] = fragments_count
        return result

    # ------------------------------------------------------------------
    # Empty result
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(query: str) -> Dict[str, Any]:
        return {
            "answer": "",
            "fragments": [],
            "confidence": 0.0,
            "label": "none",
            "query": query,
            "method": "no_results",
        }


# ======================================================================
# Self-test
# ======================================================================

def _self_test() -> None:
    """Run: python -m hms.scripts.reconstructive_recall"""
    import tempfile

    print("Testing ReconstructiveRecaller ...")

    rr = ReconstructiveRecaller({})

    # Test empty query
    empty = rr.recall("", {})
    assert empty["label"] == "none"
    assert empty["method"] == "no_results"
    print("[empty query] OK")

    # Test tag confidence
    tagged_exact = rr._tag_confidence({"confidence": 0.9, "fragments_count": 1})
    assert tagged_exact["label"] == "exact"
    tagged_recon = rr._tag_confidence({"confidence": 0.6})
    assert tagged_recon["label"] == "reconstructed"
    tagged_infer = rr._tag_confidence({"confidence": 0.3})
    assert tagged_infer["label"] == "inferred"
    print("[confidence tags] OK")

    # Test local keyword search (should find something if beliefs.json exists)
    frags = rr._search_fragments("test", top_k=3)
    assert isinstance(frags, list)
    print(f"[fragment search] found {len(frags)} fragments")

    # Test LLM synthesis fallback (no LLM available in test)
    synthesis = rr._llm_synthesize(
        [{"text": "用户喜欢编程", "score": 0.7}],
        "你喜欢什么？",
    )
    assert "answer" in synthesis
    print(f"[synthesis fallback] OK (answer={synthesis['answer'][:50]}...)")

    print("✓ All ReconstructiveRecaller self-tests passed.")


if __name__ == "__main__":
    _self_test()
