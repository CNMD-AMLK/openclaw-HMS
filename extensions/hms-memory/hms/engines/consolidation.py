"""
HMS v5 — Consolidation Engine (adapted from hms-v4 scripts/consolidation.py).
"""

from __future__ import annotations
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hms.utils.llm import LLMAnalyzer

logger = logging.getLogger("hms.consolidation")

_EMOTION_PATTERNS = [
    (re.compile(r'(太|很|非常|特别|超级|极其)\s*(高兴|开心|兴奋|激动|满意|舒服|棒|爽|快乐|幸福|感动|惊喜)'), "high", True),
    (re.compile(r'(太|很|非常|特别|超级|极其)\s*(愤怒|生气|烦躁|焦虑|紧张|害怕|恐惧|悲伤|难过|伤心|失望|痛苦|绝望|恶心|讨厌|烦|郁闷|无聊|孤独|寂寞|疲惫|累|困)'), "high", False),
    (re.compile(r'(!|！){2,}'), "high", None),
    (re.compile(r'(高兴|开心|兴奋|满意|舒服|棒|爽|快乐|幸福|感动|惊喜)'), "positive", True),
    (re.compile(r'(愤怒|生气|烦躁|焦虑|紧张|害怕|恐惧|悲伤|难过|伤心|失望|痛苦|绝望|恶心|讨厌|烦|郁闷|无聊|孤独|寂寞|疲惫|累|困)'), "negative", False),
]


class ConsolidationEngine:
    def __init__(self, config: Dict[str, Any], *, llm: Optional[LLMAnalyzer] = None) -> None:
        self.cfg = config
        self.llm = llm or LLMAnalyzer(config)
        self._embed_cache = None

    def set_embed_cache(self, cache: Any) -> None:
        self._embed_cache = cache

    def run(self, memories: List[Dict[str, Any]], report: Dict[str, Any]) -> None:
        """Run consolidation on memories: compress similar ones, update fingerprints."""
        if not memories:
            return

        # 1. Compress similar memories (simple embedding-based deduplication)
        try:
            compressed = self._compress_memories(memories)
            report["compressed"] = compressed
        except Exception as e:
            report["errors"].append(f"compression: {e}")

        # 2. Update importance based on recency and access
        try:
            updated = self._update_importance(memories)
            report["updated"] = updated
        except Exception as e:
            report["errors"].append(f"importance_update: {e}")

    def _compress_memories(self, memories: List[Dict]) -> int:
        """Merge highly similar memories using embedding similarity."""
        if not self._embed_cache or len(memories) < 2:
            return 0

        compressed = 0
        i = 0
        while i < len(memories):
            mem_a = memories[i]
            j = i + 1
            while j < len(memories):
                sim = self._embed_cache.similarity(mem_a.get("text", ""), memories[j].get("text", ""))
                if sim >= 0.95:
                    # Merge: keep higher importance, update access
                    new_imp = max(mem_a.get("importance", 5), memories[j].get("importance", 5))
                    mem_a["importance"] = new_imp
                    mem_a["access_count"] = max(mem_a.get("access_count", 0), memories[j].get("access_count", 0))
                    memories.pop(j)
                    compressed += 1
                else:
                    j += 1
            i += 1
        return compressed

    def _update_importance(self, memories: List[Dict]) -> int:
        """Boost importance of frequently accessed memories."""
        updated = 0
        now = datetime.now(timezone.utc)
        for mem in memories:
            access_count = mem.get("access_count", 0)
            if access_count > 3:
                current_imp = mem.get("importance", 5)
                boost = min(access_count * 0.2, 3.0)
                new_imp = min(int(current_imp + boost), 10)
                if new_imp != current_imp:
                    mem["importance"] = new_imp
                    updated += 1
        return updated
