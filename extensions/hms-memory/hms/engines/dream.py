"""
HMS v5 — Dream Engine (adapted from hms-v4 scripts/dream_engine.py).
"""

from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hms.utils.file_utils import atomic_write_json

logger = logging.getLogger("hms.dream")


class DreamEngine:
    """
    Offline insight discovery through memory association.
    Works on memory clusters to find latent patterns.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self._insights_dir = os.path.join(config.get("dataDir", "data"), "insights")
        os.makedirs(self._insights_dir, exist_ok=True)

    def run_dream_cycle(self) -> Dict[str, Any]:
        """Run one dream cycle: load memories, find patterns, save insights."""
        # Memories are loaded by the caller (MemoryManager)
        # This is called via RPC so we can't access the adapter directly here
        return {"insights_saved": 0, "method": "dream_cycle", "note": "Call via manager for full cycle"}

    def analyze_cluster(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze a cluster of related memories for latent patterns."""
        if len(memories) < 2:
            return []

        insights = []
        texts = [m.get("text", "")[:300] for m in memories]
        combined = "\n---\n".join(texts)

        insight = {
            "type": "association",
            "related_count": len(memories),
            "preview": combined[:100],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        insights.append(insight)
        return insights

    def save_insight(self, insight: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save an insight to disk."""
        if filename is None:
            filename = f"insight_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(self._insights_dir, filename)
        atomic_write_json(path, insight)
        return path
