"""
HMS v3 — Perception Engine.

LLM-driven perception analysis. Falls back to lightweight heuristics.
Replaces v1's dictionary-based approach entirely.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .llm_analyzer import LLMAnalyzer


class PerceptionEngine:
    """
    Analyzes conversation turns using LLM for deep understanding.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self.llm = LLMAnalyzer(self.cfg)
        self._mode = self.cfg.get("llm_perception_mode", "full")

    def analyze(
        self,
        user_message: str,
        assistant_reply: str = "",
        *,
        force_llm: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze a conversation turn.

        Modes:
          - "full": always try LLM, fallback to heuristic
          - "lite": heuristic only (for high-throughput / sync path)
          - "llm_only": LLM only, return None-like on failure
        """
        if self._mode == "lite" and not force_llm:
            result = self.llm.fallback_perceive(user_message, assistant_reply)
            result["analysis_method"] = "heuristic"
            result["text_for_store"] = self._build_store_text(
                user_message, assistant_reply, result
            )
            return result

        if not force_llm:
            result = self.llm.fallback_perceive(user_message, assistant_reply)
            result["analysis_method"] = "heuristic"
            result["text_for_store"] = self._build_store_text(
                user_message, assistant_reply, result
            )
            return result

        # LLM-only path (async / process_pending)
        result = self.llm.perceive(user_message, assistant_reply)
        if result:
            result["analysis_method"] = "llm"
            result["text_for_store"] = self._build_store_text(
                user_message, assistant_reply, result
            )
            return result

        return {
            "should_remember": False,
            "analysis_method": "failed",
            "text_for_store": user_message,
        }

    @staticmethod
    def _build_store_text(
        user_message: str, assistant_reply: str, analysis: Dict[str, Any]
    ) -> str:
        """Build the text that will be stored in memory."""
        parts = [f"用户: {user_message}"]
        if assistant_reply:
            parts.append(f"助手: {assistant_reply[:300]}")

        # Append key facts if available
        key_facts = analysis.get("key_facts", [])
        if key_facts:
            parts.append(f"关键事实: {'; '.join(key_facts[:3])}")

        return "\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        return self.llm.get_stats()


# ======================================================================
# Self-test
# ======================================================================


def _self_test():
    """Run: python -m hms.scripts.perception"""
    engine = PerceptionEngine({"llm_perception_mode": "lite"})

    # Test lite mode (heuristic only)
    result = engine.analyze("我决定用 Python 来做这个项目", "好的，明白了")
    assert result["analysis_method"] == "heuristic"
    assert result["importance"] >= 5
    print(f"[lite] importance={result['importance']} method={result['analysis_method']}")

    # Test with full mode but fallback (no LLM available)
    engine2 = PerceptionEngine({"llm_perception_mode": "full"})
    result2 = engine2.analyze("帮我创建一个新的 Docker 容器")
    assert result2["intent"]["primary"] == "请求"
    print(f"[full-fallback] intent={result2['intent']['primary']}")

    print("✓ All self-tests passed.")


if __name__ == "__main__":
    _self_test()
