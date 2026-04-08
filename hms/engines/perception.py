"""
HMS v5 — Perception Engine (adapted from hms-v4 scripts/perception.py).
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from hms.utils.llm import LLMAnalyzer
from hms.utils.text import sanitize_text


class PerceptionEngine:
    def __init__(self, config: Dict[str, Any], *, llm: Optional[LLMAnalyzer] = None) -> None:
        self.cfg = config
        self.llm = llm or LLMAnalyzer(config)
        self._mode = config.get("perceptionMode", config.get("llm_perception_mode", "lite"))

    def analyze(self, user_message: str, assistant_reply: str = "",
                *, force_llm: bool = False, force_heuristic: bool = False) -> Dict[str, Any]:
        user_message = sanitize_text(user_message)
        if assistant_reply:
            assistant_reply = sanitize_text(assistant_reply)

        if self._mode == "lite" or force_heuristic:
            result = self.llm.fallback_perceive(user_message, assistant_reply)
            result["analysis_method"] = "heuristic"
            result["text_for_store"] = self._build_store_text(user_message, assistant_reply, result)
            return result

        result = self.llm.perceive(user_message, assistant_reply)
        if result:
            result["analysis_method"] = "llm"
            result["text_for_store"] = self._build_store_text(user_message, assistant_reply, result)
            return result

        result = self.llm.fallback_perceive(user_message, assistant_reply)
        result["analysis_method"] = "heuristic_fallback"
        result["text_for_store"] = self._build_store_text(user_message, assistant_reply, result)
        return result

    @staticmethod
    def _build_store_text(user_message: str, assistant_reply: str, analysis: Dict[str, Any]) -> str:
        parts = [f"用户: {user_message}"]
        if assistant_reply:
            parts.append(f"助手: {assistant_reply[:300]}")
        key_facts = analysis.get("key_facts", [])
        if key_facts:
            parts.append(f"关键事实: {'; '.join(key_facts[:3])}")
        return "\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        return self.llm.get_stats()
