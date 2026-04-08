"""
HMS v5 — LLM analyzer (adapted from hms-v4 scripts/llm_analyzer.py).
Config-driven, no Config singleton, no .env loading.
"""

from __future__ import annotations
import json
import os
import re
import threading
import time
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from .text import estimate_tokens

logger = logging.getLogger("hms.llm")


class LLMAnalyzer:
    """
    Central LLM analysis hub using OpenClaw Gateway.
    Config-driven: all settings come from the config dict passed at init.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self._timeout = config.get("llmTimeoutSeconds", 30)
        self._max_retries = config.get("llmMaxRetries", 3)
        self._call_count = 0
        self._token_count = 0
        self._budget = config.get("tokenBudgetDaily", 50000)
        self._budget_date = datetime.now(timezone.utc).date()

        self._gateway_url = config.get("gatewayUrl", os.environ.get("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:18789"))
        self._gateway_token = config.get("gatewayToken", os.environ.get("OPENCLAW_GATEWAY_TOKEN", ""))
        self._model = config.get("llmModel", "openclaw")
        self._session = requests.Session()
        if self._gateway_token:
            self._session.headers["Authorization"] = f"Bearer {self._gateway_token}"

        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._circuit_failure_threshold = 5
        self._circuit_cooldown_seconds = 300

    @staticmethod
    def _midnight_utc() -> float:
        now = time.time()
        return now - (now % 86400)

    def _check_budget(self) -> bool:
        today = datetime.now(timezone.utc).date()
        if self._budget_date != today:
            self._call_count = 0
            self._token_count = 0
            self._budget_date = today
        return self._token_count < self._budget

    def _call_llm(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> Optional[str]:
        if not self._check_budget():
            logger.warning("LLM budget exhausted, skipping call")
            return None

        now = time.time()
        if now < self._circuit_open_until:
            return None

        for attempt in range(self._max_retries):
            try:
                result = self._try_gateway_api(prompt, max_tokens, temperature)
                if result:
                    self._call_count += 1
                    self._token_count += estimate_tokens(prompt)
                    self._token_count += estimate_tokens(result)
                    self._consecutive_failures = 0
                    return result
                return None
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status == 429:
                    wait = min(5 * (2 ** attempt), 60)
                    time.sleep(wait)
                    self._consecutive_failures += 1
                elif status >= 500:
                    time.sleep(min(2 ** attempt, 30))
                    self._consecutive_failures += 1
                elif status in (401, 403):
                    self._circuit_open_until = time.time() + self._circuit_cooldown_seconds
                    return None
                else:
                    self._consecutive_failures += 1
                    if attempt < self._max_retries - 1:
                        time.sleep(min(2 ** attempt, 10))
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                time.sleep(min(2 ** attempt, 30))
                self._consecutive_failures += 1
            except Exception:
                self._consecutive_failures += 1
                if attempt < self._max_retries - 1:
                    time.sleep(min(2 ** attempt, 30))

            if self._consecutive_failures >= self._circuit_failure_threshold:
                self._circuit_open_until = time.time() + self._circuit_cooldown_seconds
                return None
        return None

    def _try_gateway_api(self, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = self._session.post(
            f"{self._gateway_url}/v1/chat/completions",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        message = choices[0].get("message", {})
        content = message.get("content") or message.get("reasoning")
        return content.strip() if content else None

    @staticmethod
    def _parse_json_response(raw: str, required_keys: Optional[set] = None) -> Optional[Dict[str, Any]]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    result = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return None
            else:
                return None
        if not isinstance(result, dict):
            return None
        if required_keys:
            missing = required_keys - set(result.keys())
            if missing:
                logger.debug("Missing required keys: %s", missing)
                return None
        return result

    def perceive(self, user_message: str, assistant_reply: str = "") -> Optional[Dict[str, Any]]:
        prompt = self._build_perceive_prompt(user_message, assistant_reply)
        raw = self._call_llm(prompt, max_tokens=800, temperature=0.1)
        if not raw:
            return None
        return self._parse_json_response(raw, required_keys={"entities", "emotion", "intent", "importance", "category"})

    def _build_perceive_prompt(self, user_message: str, assistant_reply: str) -> str:
        return f"""分析以下对话，提取关键信息：

用户消息: {user_message}
助手回复: {assistant_reply}

请以JSON格式返回：
{{
  "entities": [{"name": "实体名", "type": "类型"}],
  "emotion": {{"valence": 0.0, "arousal": 0.0, "primary_emotion": "neutral"}},
  "intent": {{"primary": "提问|请求|陈述|指令", "confidence": 0.5}},
  "importance": 1-10,
  "importance_reason": "原因",
  "category": "fact|preference|decision|procedure",
  "topics": ["话题1"],
  "key_facts": ["关键事实"],
  "should_remember": true|false
}}"""

    @staticmethod
    def fallback_perceive(user_message: str, assistant_reply: str = "") -> Dict[str, Any]:
        """Heuristic fallback when LLM is unavailable."""
        msg = user_message.strip()
        importance = 4
        for s in ["决定", "必须", "永远", "关键", "核心"]:
            if s in user_message:
                importance = 9
                break
        else:
            for s in ["计划", "方案", "确认", "重要"]:
                if s in user_message:
                    importance = 7
                    break

        intent = "陈述"
        if "?" in user_message or "？" in user_message:
            intent = "提问"
        elif re.match(r"^(帮我|请|能不能|能否|可以|麻烦)", msg):
            intent = "请求"
        elif re.match(r"^(执行|运行|创建|删除|打开|关闭)", msg):
            intent = "指令"

        category = "fact"
        if any(w in user_message for w in ["我喜欢", "我偏好", "我习惯"]):
            category = "preference"
        elif any(w in user_message for w in ["决定", "选择", "确认"]):
            category = "decision"

        return {
            "entities": [],
            "emotion": {"valence": 0.0, "arousal": 0.15, "primary_emotion": "neutral"},
            "intent": {"primary": intent, "confidence": 0.5},
            "importance": importance,
            "importance_reason": "fallback heuristic",
            "category": category,
            "topics": [],
            "key_facts": [user_message[:100]] if importance >= 6 else [],
            "should_remember": importance >= 6,
        }

    def get_stats(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "call_count": self._call_count,
            "token_count_estimate": self._token_count,
            "budget_remaining": max(0, self._budget - self._token_count),
            "consecutive_failures": self._consecutive_failures,
            "circuit_open": now < self._circuit_open_until,
        }
