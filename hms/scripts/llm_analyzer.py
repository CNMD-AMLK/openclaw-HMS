"""
HMS — LLM Analyzer Core.

Replaces all dictionary-based analysis with LLM calls.
Falls back to lightweight heuristics when LLM is unavailable.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import logging
from datetime import datetime, timezone

import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import estimate_tokens

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """
    Central LLM analysis hub. All cognitive tasks route through here.

    Uses the current OpenClaw model via the Gateway API.
    """

    _PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
    _env_loaded = False
    _env_lock = threading.Lock()

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        from .config_loader import Config
        self.cfg = config or Config.get()
        self._model = self.cfg.get("llm_model", "__current__")
        self._timeout = self.cfg.get("llm_timeout_seconds", 30)
        self._max_retries = self.cfg.get("llm_max_retries", 3)
        self._call_count = 0
        self._token_count = 0
        self._budget = self.cfg.get("llm_budget_tokens_per_day", 50000)
        self._budget_reset = self._midnight_utc()

        # Rate limiting
        self._call_timestamps: list[float] = []
        self._call_timestamps_lock = threading.Lock()
        self._rate_limit_per_minute = self.cfg.get("llm_rate_limit_per_minute", 10)
        self._rate_limit_per_hour = self.cfg.get("llm_rate_limit_per_hour", 30)

        # Gateway configuration
        self._gateway_url = self.cfg.get("gateway_url", "http://127.0.0.1:18789")
        self._gateway_token = self.cfg.get("gateway_token", "")
        self._model = self.cfg.get("llm_model", "openclaw")
        self._session = requests.Session()
        # Set auth header if token is provided
        if self._gateway_token:
            self._session.headers["Authorization"] = f"Bearer {self._gateway_token}"

        # Load .env file if not already loaded (thread-safe)
        if not LLMAnalyzer._env_loaded:
            with LLMAnalyzer._env_lock:
                if not LLMAnalyzer._env_loaded:
                    LLMAnalyzer._load_env()

        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._circuit_failure_threshold = 5
        self._circuit_cooldown_seconds = 300  # 5 minutes

        # Persisted circuit breaker state
        cache_dir = Path(self.cfg.get("cache_dir", "cache"))
        if not cache_dir.is_absolute():
            cache_dir = Path(__file__).parent.parent / 'cache'
        self._cb_state_path = str(cache_dir / 'circuit_breaker.json')
        self._load_circuit_breaker_state()

    def close(self) -> None:
        """Close the HTTP session and release connections."""
        if self._session:
            self._session.close()
            self._session = None

    @staticmethod
    def _midnight_utc() -> float:
        """Return the Unix timestamp for the most recent midnight UTC."""
        now = time.time()
        return now - (now % 86400)

    # FIX: whitelist of allowed env keys to prevent malicious overrides
    _ALLOWED_ENV_KEYS = {
        "OPENCLAW_GATEWAY_URL",
        "HMS_LLM_MODEL",
        "HMS_GATEWAY_URL",
        "HMS_GATEWAY_TOKEN",
    }

    @classmethod
    def _load_env(cls) -> None:
        """Load .env file if exists, with whitelist protection."""
        # Try multiple paths: package root, cwd, ~/.hms/
        candidates = [
            Path(__file__).parent.parent.parent / ".env",
            Path.cwd() / ".env",
            Path.home() / ".hms" / ".env",
        ]
        env_path = None
        for p in candidates:
            if p.exists():
                env_path = p
                break
        if env_path is None:
            cls._env_loaded = True
            return
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    if key in cls._ALLOWED_ENV_KEYS:
                        os.environ.setdefault(key, val.strip())
        cls._env_loaded = True

    def _load_circuit_breaker_state(self) -> None:
        """Load persisted circuit breaker state if available."""
        if not self._cb_state_path or not os.path.isfile(self._cb_state_path):
            return
        try:
            with open(self._cb_state_path, "r") as f:
                data = json.load(f)
            self._consecutive_failures = data.get("consecutive_failures", 0)
            self._circuit_open_until = data.get("circuit_open_until", 0.0)
            logger.debug("Loaded circuit breaker state: failures=%d open_until=%.0f",
                         self._consecutive_failures, self._circuit_open_until)
        except (json.JSONDecodeError, IOError):
            pass

    def _save_circuit_breaker_state(self) -> None:
        """Persist circuit breaker state to disk."""
        if not self._cb_state_path:
            return
        try:
            os.makedirs(os.path.dirname(self._cb_state_path), exist_ok=True)
            from .file_utils import atomic_write_json
            atomic_write_json(self._cb_state_path, {
                "consecutive_failures": self._consecutive_failures,
                "circuit_open_until": self._circuit_open_until,
            })
        except IOError as e:
            logger.debug("Failed to save circuit breaker state: %s", e)

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
        Call the current OpenClaw model via Gateway with exponential backoff,
        error classification, and circuit breaker.
        """
        # Budget check — FIX: use date-based reset instead of time-delta
        today = datetime.now(timezone.utc).date()
        if not hasattr(self, "_budget_date") or self._budget_date != today:
            self._call_count = 0
            self._token_count = 0
            self._budget_date = today

        if self._token_count >= self._budget:
            logger.warning(
                "LLM budget exhausted (%d/%d tokens). Skipping call, falling back to heuristic.",
                self._token_count, self._budget,
            )
            return None

        # Circuit breaker check
        now = time.time()
        if now < self._circuit_open_until:
            logger.debug("Circuit breaker open, skipping LLM call (reopens in %.0fs)",
                         self._circuit_open_until - now)
            return None

        # Rate limit check: per-minute and per-hour
        if self._check_rate_limit(now) is not None:
            return None

        for attempt in range(self._max_retries):
            try:
                result = self._try_gateway_api(prompt, max_tokens, temperature)
                if result:
                    # Record call timestamp for rate limiting
                    with self._call_timestamps_lock:
                        self._call_timestamps.append(now)
                    self._call_count += 1
                    prompt_tokens = estimate_tokens(prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False))
                    self._token_count += prompt_tokens
                    self._token_count += estimate_tokens(result)
                    # Reset circuit breaker on success
                    self._consecutive_failures = 0
                    self._save_circuit_breaker_state()
                    return result
                # Empty result — not a failure, just unavailable
                return None
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0

                if status == 429:
                    # Rate limited: read Retry-After header
                    retry_after = self._parse_retry_after(exc.response)
                    wait = min(retry_after, 60)
                    time.sleep(wait)
                    self._consecutive_failures += 1
                elif status >= 500:
                    # Server error: exponential backoff
                    wait = min(2 ** attempt, 30)
                    time.sleep(wait)
                    self._consecutive_failures += 1
                elif status == 401 or status == 403:
                    # Auth error: no point retrying
                    logger.error("Gateway auth error (HTTP %d), opening circuit", status)
                    self._trip_circuit()
                    return None
                else:
                    # Other client errors
                    self._consecutive_failures += 1
                    if attempt < self._max_retries - 1:
                        time.sleep(min(2 ** attempt, 30))

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                # Network issues: exponential backoff
                wait = min(2 ** attempt, 30)
                time.sleep(wait)
                self._consecutive_failures += 1

            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.debug("LLM call failed: %s", e)
                self._consecutive_failures += 1
                if attempt < self._max_retries - 1:
                    time.sleep(min(2 ** attempt, 30))

            # Check if circuit should trip
            if self._consecutive_failures >= self._circuit_failure_threshold:
                self._trip_circuit()
                return None

        return None

    def _trip_circuit(self) -> None:
        """Open the circuit breaker — pause all LLM calls."""
        self._circuit_open_until = time.time() + self._circuit_cooldown_seconds
        self._save_circuit_breaker_state()
        logger.warning(
            "Circuit breaker tripped after %d consecutive failures. Cooldown: %ds",
            self._consecutive_failures, self._circuit_cooldown_seconds,
        )

    def _check_rate_limit(self, now: float) -> Optional[str]:
        """Check per-minute and per-hour rate limits. Returns error message if exceeded."""
        with self._call_timestamps_lock:
            # Remove timestamps older than 1 hour
            one_hour_ago = now - 3600
            self._call_timestamps = [
                t for t in self._call_timestamps if t > one_hour_ago
            ]
            # Count calls in the last minute and hour
            one_min_ago = now - 60
            calls_last_min = sum(1 for t in self._call_timestamps if t > one_min_ago)
            calls_last_hour = len(self._call_timestamps)

            if calls_last_min >= self._rate_limit_per_minute:
                logger.warning(
                    "Rate limit: %d calls in last minute (limit=%d). Skipping.",
                    calls_last_min, self._rate_limit_per_minute,
                )
                return "rate_limit_per_minute"

            if calls_last_hour >= self._rate_limit_per_hour:
                logger.warning(
                    "Rate limit: %d calls in last hour (limit=%d). Skipping.",
                    calls_last_hour, self._rate_limit_per_hour,
                )
                return "rate_limit_per_hour"

        return None

    @staticmethod
    def _parse_retry_after(response: Optional[requests.Response]) -> float:
        """Parse Retry-After header from a 429 response."""
        if response is None:
            return 5.0
        raw = response.headers.get("Retry-After", "")
        if not raw:
            return 5.0
        try:
            return float(raw)
        except ValueError:
            # Could be HTTP-date format, default to 5s
            return 5.0

    def _try_gateway_api(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> Optional[str]:
        """Call OpenClaw Gateway API instead of direct model API."""
        # Use Gateway's chat completion endpoint
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
        try:
            data = resp.json()
        except json.JSONDecodeError:
            logger.warning("Gateway returned non-JSON response: %s", resp.text[:200])
            return None

        # Safely extract message from response
        choices = data.get("choices", [])
        if not choices:
            logger.warning("Empty choices in API response")
            return None

        message = choices[0].get("message", {})
        content = message.get("content")
        if content:
            return content.strip()
        # Some models may return reasoning instead of content
        reasoning = message.get("reasoning")
        if reasoning:
            return reasoning.strip()
        return None

    # ==================================================================
    # Health check
    # ==================================================================

    def health_check(self) -> Dict[str, Any]:
        """Verify Gateway connectivity and API availability."""
        result = {
            "gateway_reachable": False,
            "chat_api_available": False,
            "gateway_url": self._gateway_url,
            "errors": [],
        }
        # Check gateway reachability
        try:
            resp = self._session.get(
                f"{self._gateway_url}/health",
                timeout=5,
            )
            if resp.status_code == 200:
                result["gateway_reachable"] = True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            result["errors"].append(f"Gateway unreachable: {e}")
            return result

        # Check chat API
        try:
            resp = self._session.post(
                f"{self._gateway_url}/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 10,
                },
                timeout=10,
            )
            if resp.status_code in (200, 400, 422):
                result["chat_api_available"] = True
            else:
                result["errors"].append(f"Chat API returned HTTP {resp.status_code}")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            result["errors"].append(f"Chat API error: {e}")

        return result

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
        return self._parse_json_response(raw, required_keys=self._PERCEPTION_SCHEMA)

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
        for i, mem in enumerate(existing_memories):
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
        return self._parse_json_response(raw, required_keys=self._COLLIDE_SCHEMA)

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
        Uses regex patterns for more robust intent detection.
        """
        text = user_message + " " + assistant_reply
        msg = user_message.strip()

        importance = 4
        high_signals = ["决定", "必须", "永远", "关键", "核心", "禁忌"]
        mid_signals = ["计划", "方案", "确认", "重要", "任务"]

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
                casual_patterns = re.compile(
                    r'^(嗯|好的|行|哈哈|ok|OK|哦|好|知道了|明白|了解了)',
                    re.IGNORECASE,
                )
                if casual_patterns.match(msg) and len(msg) <= 8:
                    importance = 2

        intent = "陈述"
        if "?" in user_message or "？" in user_message:
            intent = "提问"
        elif re.match(r'^(帮我|请|能不能|能否|可以|麻烦|帮我一下)', msg):
            intent = "请求"
        elif re.match(r'^(执行|运行|创建|删除|打开|关闭|安装|卸载)', msg):
            intent = "指令"

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

    # Schema validation rules for top-level expected keys
    _PERCEPTION_SCHEMA = {"entities", "emotion", "intent", "importance", "category", "topics", "key_facts", "should_remember"}
    _COLLIDE_SCHEMA = {"contradictions", "reinforcements", "associations", "inferences"}
    _GENERIC_MINIMAL_SCHEMA = {}  # no requirement for unknown cognitive tasks

    @staticmethod
    def _parse_json_response(raw: str, required_keys: Optional[set] = None) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, handling markdown code blocks.

        Args:
            raw: Raw LLM text response
            required_keys: Optional set of top-level keys that must exist;
                           if provided, missing keys cause validation failure.

        Returns:
            Parsed dict on success, None if unparseable or schema invalid.
        """
        text = raw.strip()

        # Strip markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    result = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            else:
                return None

        # Validate type
        if not isinstance(result, dict):
            logger.debug("Parsed LLM response is not a dict: %s", type(result))
            return None

        # Validate required keys if provided
        if required_keys:
            missing = required_keys - set(result.keys())
            if missing:
                logger.debug("LLM response missing required keys: %s", missing)
                return None

        return result

    # ==================================================================
    # Stats
    # ==================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        now = time.time()
        circuit_open = now < self._circuit_open_until
        return {
            "call_count": self._call_count,
            "token_count_estimate": self._token_count,
            "budget_remaining": max(0, self._budget - self._token_count),
            "budget_reset_in_hours": max(
                0, (86400 - (now - self._budget_reset)) / 3600
            ),
            "consecutive_failures": self._consecutive_failures,
            "circuit_open": circuit_open,
            "circuit_remaining_seconds": max(0, self._circuit_open_until - now) if circuit_open else 0,
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

    # Test token estimation
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("你好世界测试") > estimate_tokens("hello test")
    print(f"[token estimate] OK")

    # Test circuit breaker state
    stats = analyzer.get_stats()
    assert stats["consecutive_failures"] == 0
    assert stats["circuit_open"] is False
    print(f"[circuit breaker] OK")

    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
