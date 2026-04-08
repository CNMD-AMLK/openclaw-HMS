"""
HMS v5 — Creative Associator (adapted from hms-v4 scripts/creative_assoc.py).
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List

from hms.utils.llm import LLMAnalyzer

logger = logging.getLogger("hms.creative")


class CreativeAssociator:
    """
    Generate creative insights by finding unexpected connections
    between memories from different domains.
    """

    def __init__(self, config: Dict[str, Any], *, llm: Optional[LLMAnalyzer] = None) -> None:
        self.cfg = config
        self.llm = llm or LLMAnalyzer(config)

    def generate_insights(self) -> Dict[str, Any]:
        """Generate creative insights from memory pool."""
        return {"insights_count": 0, "method": "creative", "note": "Call via manager for full cycle"}

    def find_cross_domain_links(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find unexpected connections between memories from different categories."""
        if len(memories) < 3:
            return []

        categories = {}
        for mem in memories:
            cat = mem.get("category", "fact")
            categories.setdefault(cat, []).append(mem)

        if len(categories) < 2:
            return []

        links = []
        cats = list(categories.keys())
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                cat_a, cat_b = cats[i], cats[j]
                # Simple: pick one from each
                link = {
                    "type": "cross_domain",
                    "from_category": cat_a,
                    "to_category": cat_b,
                    "from_text": categories[cat_a][0].get("text", "")[:100],
                    "to_text": categories[cat_b][0].get("text", "")[:100],
                }
                links.append(link)
        return links
