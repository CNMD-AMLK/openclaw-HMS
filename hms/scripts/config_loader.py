"""统一配置加载器"""
import json
import os
from pathlib import Path


class Config:
    """单例配置"""
    _instance = None
    _data = None

    @classmethod
    def get(cls):
        if cls._data is None:
            cls._data = cls._load()
        return cls._data

    @classmethod
    def _load(cls):
        config_path = Path(__file__).parent.parent / "config.json"
        with open(config_path) as f:
            data = json.load(f)
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
        cls._data = cls._load()
