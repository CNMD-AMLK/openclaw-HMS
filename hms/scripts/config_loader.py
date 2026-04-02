"""统一配置加载器"""
import json
import os
import threading
from pathlib import Path


class Config:
    """单例配置"""
    _instance = None
    _data = None
    _lock = threading.Lock()

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
        try:
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {e}")
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
