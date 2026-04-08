"""
HMS v5 — Reconstructive Recall Engine (adapted from hms-v4 scripts/reconstructive_recall.py).
Uses embedding similarity for fragment search, then reconstructs via LLM.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional

from hms.utils.llm import LLMAnalyzer

logger = logging.getLogger("hms.recall")


class ReconstructiveRecaller:
    def __init__(self, config: Dict[str, Any], *, llm: Optional[LLMAnalyzer] = None) -> None:
        self.cfg = config
        self.llm = llm or LLMAnalyzer(config)

    def recall(self, query: str, perception: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """
        Reconstructive recall: use LLM to synthesize a narrative answer
        from memory fragments, not just retrieve them.
        """
        memories = self._search_fragments(query, top_k=top_k * 2)

        if not memories:
            return {"method": "no_memories", "query": query, "count": 0, "memories": []}

        mem_lines = []
        for i, mem in enumerate(memories):
            text = mem.get("text", "")[:200]
            importance = mem.get("importance", 5)
            mem_lines.append(f"[记忆{i+1} (重要性:{importance})] {text}")

        prompt = f"""基于以下记忆碎片，回答用户问题。如果没有相关信息，请明确说明不知道。

用户问题: {query}

记忆碎片:
{chr(10).join(mem_lines)}

请先列出相关记忆，然后给出综合回答。如果记忆中有矛盾，给出最可靠的版本并注明矛盾。
回答格式：
相关记忆：[列出]
综合回答：[你的回答]"""

        raw = self.llm._call_llm(prompt, max_tokens=800, temperature=0.3)
        return {
            "method": "reconstructive",
            "query": query,
            "count": len(memories),
            "memories": memories[:top_k],
            "answer": raw or "无法回忆相关记忆。",
        }

    def _search_fragments(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for memory fragments relevant to query.
        This is called from MemoryManager which has the adapter.
        """
        # The actual search is done by the adapter passed from manager
        # This method is a placeholder for the interface
        return []


def simple_recall(adapter, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Simple recall using the adapter's semantic search."""
    return adapter.recall(query=query, top_k=top_k)
