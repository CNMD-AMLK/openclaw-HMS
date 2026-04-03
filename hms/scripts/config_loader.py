"""统一配置加载器 — v3.6.3: 优雅降级"""
import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """单例配置 — 配置加载失败时返回默认值，不阻断启动"""
    _instance = None
    _data = None
    _lock = threading.Lock()

    _DEFAULTS = {
        "gateway_url": "http://127.0.0.1:18789",
        "gateway_token": "",
        "llm_model": "openclaw",
        "cache_dir": "cache",
        "importance_threshold": 6,
        "retrieval_top_k": 30,
        "forget_base_threshold": 0.08,
        "emotion_decay_slowdown_factor": 3.0,
        "llm_budget_tokens_per_day": 50000,
    }

    @classmethod
    def get(cls):
        if cls._data is None:
            with cls._lock:
                if cls._data is None:
                    cls._data = cls._load()
        return cls._data

    @classmethod
    def _load(cls):
        config_path = Path(__file__).parent.parent / "config.json"
        data = dict(cls._DEFAULTS)
        try:
            with open(config_path, encoding="utf-8") as f:
                file_data = json.load(f)
            data.update(file_data)
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(
                "Config load failed from %s (%s), using defaults. "
                "HMS will start with default configuration.",
                config_path, e,
            )
        # 从环境变量覆盖敏感配置
        data["gateway_token"] = os.environ.get(
            "HMS_GATEWAY_TOKEN",
            data.get("gateway_token", ""),
        )
        data["gateway_url"] = os.environ.get(
            "HMS_GATEWAY_URL",
            data.get("gateway_url", "http://127.0.0.1:18789"),
        )
        return data

    @classmethod
    def reload(cls):
        with cls._lock:
            cls._data = cls._load()
