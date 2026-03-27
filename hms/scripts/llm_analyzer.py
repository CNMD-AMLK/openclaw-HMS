"""
HMS v2 — LLM Analyzer Core.

Replaces all dictionary-based analysis with LLM calls.
Falls back to lightweight heuristics when LLM is unavailable.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time

import requests
from pathlib import Path
from typing import Any, Dict, List, Optional


class LLMAnalyzer:
    """
    Central LLM analysis hub. All cognitive tasks route through here.

    Uses the current OpenClaw model via the `openclaw` CLI or a direct
    Python subprocess call to the model's API.
    """

    _PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self._model = self.cfg.get("llm_model", "__current__")
        self._timeout = self.cfg.get("llm_timeout_seconds", 30)
        self._max_retries = self.cfg.get("llm_max_retries", 2)
        self._call_count = 0
        self._token_count = 0
        self._budget = self.cfg.get("llm_budget_tokens_per_day", 50000)
        self._budget_reset = time.time()

    # ==================================================================
    # Core LLM call
    # ==================================================================

    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> Optional[str]:
        """
        Call the current OpenClaw model. Tries multiple approaches:
        1. OpenClaw CLI (openclaw chat)
        2. Python subprocess with model API
        3. Fallback: return None (caller must handle)

        All calls are budget-controlled.
        """
        # Budget check
        if time.time() - self._budget_reset > 86400:
            self._call_count = 0
            self._token_count = 0
            self._budget_reset = time.time()

        if self._token_count >= self._budget:
            return None  # budget exhausted

        for attempt in range(self._max_retries):
            try:
                result = self._try_openclaw_cli(prompt, max_tokens, temperature)
                if result:
                    self._call_count += 1
                    self._token_count += len(result) // 2  # rough estimate
                    return result
            except Exception:
                if attempt < self._max_retries - 1:
                    time.sleep(1)
                    continue
        return None

    def _try_openclaw_cli(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> Optional[str]:
        """Call OpenRouter API via Python requests."""
        api_key = os.environ.get(
            "OPENROUTER_API_KEY",
            "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",
        )
        model = os.environ.get("OPENROUTER_MODEL", "xiaomi/mimo-v2-pro")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    # ==================================================================
    # Prompt loading
    # ==================================================================

    def _load_prompt(self, name: str, **kwargs: Any) -> str:
        """Load a prompt template and fill in variables."""
        prompt_path = self._PROMPTS_DIR / f"{name}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        template = prompt_path.read_text(encoding="utf-8")
        for key, value in kwargs.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template

    # ==================================================================
    # High-level analysis methods
    # ==================================================================

    def perceive(
        self, user_message: str, assistant_reply: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Full perception analysis via LLM.
        Returns structured dict or None on failure.
        """
        prompt = self._load_prompt(
            "perceive",
            user_message=user_message,
            assistant_reply=assistant_reply,
        )
        raw = self._call_llm(
            prompt,
            max_tokens=self.cfg.get("llm_perception_max_tokens", 800),
            temperature=0.1,
        )
        if not raw:
            return None
        return self._parse_json_response(raw)

    def collide(
        self,
        new_perception: Dict[str, Any],
        existing_memories: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Collision detection via LLM semantic analysis.
        """
        # Format existing memories compactly
        mem_lines = []
        for i, mem in enumerate(existing_memories[:10]):  # cap at 10
            text = mem.get("text", "")[:200]
            mem_id = mem.get("id", f"mem_{i}")
            mem_lines.append(f"[{mem_id}] {text}")
        mems_str = "\n".join(mem_lines) if mem_lines else "(无已有记忆)"

        prompt = self._load_prompt(
            "collide",
            new_perception=json.dumps(new_perception, ensure_ascii=False, indent=2),
            existing_memories=mems_str,
        )
        raw = self._call_llm(
            prompt,
            max_tokens=self.cfg.get("llm_deep_analysis_max_tokens", 1500),
            temperature=0.1,
        )
        if not raw:
            return None
        return self._parse_json_response(raw)

    def consolidate(
        self,
        conversations: List[Dict[str, str]],
        fingerprint: Dict[str, Any],
        max_tokens: int = 500,
    ) -> Optional[Dict[str, Any]]:
        """
        Compress conversations into structured summary via LLM.
        """
        # Format conversations
        conv_lines = []
        for turn in conversations:
            user = turn.get("user", "")
            assistant = turn.get("assistant", "")
            conv_lines.append(f"用户: {user}\n助手: {assistant}")
        convs_str = "\n---\n".join(conv_lines)

        prompt = self._load_prompt(
            "consolidate",
            conversations=convs_str,
            fingerprint=json.dumps(fingerprint, ensure_ascii=False, indent=2),
            max_tokens=str(max_tokens),
        )
        raw = self._call_llm(
            prompt,
            max_tokens=self.cfg.get("llm_consolidation_max_tokens", 2000),
            temperature=0.2,
        )
        if not raw:
            return None
        return self._parse_json_response(raw)

    def update_fingerprint(
        self,
        current_fingerprint: Dict[str, Any],
        new_summaries: List[Dict[str, Any]],
        max_tokens: int = 2000,
    ) -> Optional[Dict[str, Any]]:
        """
        Update cognitive fingerprint based on new conversation summaries.
        """
        summaries_str = json.dumps(new_summaries, ensure_ascii=False, indent=2)

        prompt = self._load_prompt(
            "fingerprint",
            current_fingerprint=json.dumps(
                current_fingerprint, ensure_ascii=False, indent=2
            ),
            new_summaries=summaries_str,
            max_tokens=str(max_tokens),
        )
        raw = self._call_llm(
            prompt,
            max_tokens=self.cfg.get("llm_fingerprint_max_tokens", 1500),
            temperature=0.2,
        )
        if not raw:
            return None
        return self._parse_json_response(raw)

    # ==================================================================
    # Fallback heuristics (when LLM is unavailable)
    # ==================================================================

    @staticmethod
    def fallback_perceive(user_message: str, assistant_reply: str = "") -> Dict[str, Any]:
        """
        Lightweight heuristic fallback — no LLM required.
        Much simpler than v1 but always available.
        """
        text = user_message + " " + assistant_reply

        # Simple importance scoring
        importance = 4
        high_signals = ["决定", "必须", "永远", "关键", "核心", "禁忌"]
        mid_signals = ["计划", "方案", "确认", "重要", "任务"]
        low_signals = ["嗯", "好的", "行", "哈哈", "ok", "OK"]

        for s in high_signals:
            if s in user_message:
                importance = 9
                break
        else:
            for s in mid_signals:
                if s in user_message:
                    importance = 7
                    break
            else:
                for s in low_signals:
                    if user_message.strip().startswith(s):
                        importance = 2
                        break

        # Simple intent
        intent = "陈述"
        if "?" in user_message or "？" in user_message:
            intent = "提问"
        elif any(w in user_message for w in ["帮我", "请", "能不能"]):
            intent = "请求"
        elif any(w in user_message for w in ["执行", "运行", "创建", "删除"]):
            intent = "指令"

        # Simple category
        category = "fact"
        if any(w in user_message for w in ["我喜欢", "我偏好", "我习惯"]):
            category = "preference"
        elif any(w in user_message for w in ["决定", "选择", "确认"]):
            category = "decision"

        return {
            "entities": [],
            "emotion": {
                "valence": 0.0,
                "arousal": 0.15,
                "primary_emotion": "neutral",
                "evidence": "fallback: no LLM",
            },
            "intent": {"primary": intent, "confidence": 0.5},
            "importance": importance,
            "importance_reason": "fallback heuristic",
            "category": category,
            "topics": [],
            "key_facts": [user_message[:100]] if importance >= 6 else [],
            "should_remember": importance >= 6,
        }

    # ==================================================================
    # JSON parsing helpers
    # ==================================================================

    @staticmethod
    def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = raw.strip()

        # Strip markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
        return None

    # ==================================================================
    # Stats
    # ==================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        return {
            "call_count": self._call_count,
            "token_count_estimate": self._token_count,
            "budget_remaining": max(0, self._budget - self._token_count),
            "budget_reset_in_hours": max(
                0, (86400 - (time.time() - self._budget_reset)) / 3600
            ),
        }


# ======================================================================
# Self-test
# ======================================================================


def _self_test():
    """Run: python -m hms.scripts.llm_analyzer"""
    analyzer = LLMAnalyzer({"llm_budget_tokens_per_day": 100000})

    # Test fallback (always works)
    result = analyzer.fallback_perceive("我觉得这个方案非常好！决定就用这个了。")
    assert result["importance"] >= 7
    assert result["category"] == "decision"
    print(f"[fallback] importance={result['importance']} category={result['category']}")

    # Test fallback for casual message
    result2 = analyzer.fallback_perceive("嗯好的")
    assert result2["importance"] <= 3
    print(f"[fallback casual] importance={result2['importance']}")

    # Test JSON parsing
    test_json = '```json\n{"key": "value"}\n```'
    parsed = analyzer._parse_json_response(test_json)
    assert parsed == {"key": "value"}
    print(f"[json parse] OK")

    print("✓ All self-tests passed.")


if __name__ == "__main__":
    _self_test()
