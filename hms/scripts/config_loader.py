"""Unified config loader for HMS v4.

v4 change: config.json is now minimal (~10 keys).
All secondary values are provided as hard-coded defaults here,
so no config key ever raises KeyError at runtime.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict

class Config:
    """Singleton configuration with comprehensive defaults."""

    _instance: "Config | None" = None
    _data: Dict[str, Any] | None = None
    _lock = threading.Lock()

    # ──────────────────────────────────────────────────────
    # Comprehensive defaults for every config key the codebase may read.
    # v4 user only needs to set ~10 keys in config.json; everything else
    # falls through to these sensible values.
    # ──────────────────────────────────────────────────────
    DEFAULTS: Dict[str, Any] = {
        # Core
        "cache_dir": "cache",
        "importance_threshold": 6,
        "retrieval_top_k": 30,
        "forget_base_threshold": 0.08,
        "emotion_decay_slowdown_factor": 3.0,
        "reinforcement_confidence_delta": 0.05,
        "abstraction_min_cluster_size": 3,
        "similarity_merge_threshold": 0.92,
        "dedup_similarity_threshold": 0.95,

        # Cron
        "consolidation_cron": "0 3 * * *",
        "forget_cron": "0 4 * * 0",

        # Gateway
        "gateway_url": "http://127.0.0.1:18789",
        "gateway_token": "",
        "lancedb_collection": "default",
        "graph_recall_depth": 2,

        # Context / working memory
        "working_memory_recent_turns": 80,
        "compress_every_n_turns": 50,
        "process_pending_max_batch": 50,
        "max_pending_size": 1000,

        # LLM
        "llm_model": "openclaw",
        "llm_timeout_seconds": 30,
        "llm_max_retries": 3,
        "llm_perception_mode": "lite",
        "llm_perception_max_tokens": 2000,
        "llm_deep_analysis_batch_size": 5,
        "llm_deep_analysis_max_tokens": 4000,
        "llm_consolidation_max_turns": 20,
        "llm_consolidation_max_tokens": 6000,
        "llm_fingerprint_max_tokens": 4000,
        "llm_budget_tokens_per_day": 50000,
        "llm_rate_limit_per_minute": 8,
        "llm_rate_limit_per_hour": 25,

        # Compression
        "compression_window_turns": 50,
        "compression_max_summary_tokens": 2000,

        # Timeline
        "timeline_update_interval_hours": 24,
        "timeline_max_topics": 20,
        "timeline_max_entries_per_topic": 10,

        # Fingerprint
        "fingerprint_update_interval_hours": 12,
        "fingerprint_max_tokens": 4000,

        # Agent roles
        "agent_roles": {"chat": "助手", "worker": "工作分身"},
    }

    # ── Context tier presets (used by MemoryManager._apply_tier) ──
    CONTEXT_TIERS: Dict[str, Dict[str, Any]] = {
        "32k": {
            "model_context_window": 32000,
            "context_budget": {
                "system_prompt": 2000,
                "cognitive_fingerprint_ratio": 0.05,
                "topic_timelines_ratio": 0.08,
                "compressed_summaries_ratio": 0.13,
                "injected_memories_ratio": 0.15,
                "recent_turns_ratio": 0.34,
                "response_reserve_ratio": 0.15,
                "buffer_ratio": 0.10,
            },
            "retrieval_top_k": 8,
            "working_memory_recent_turns": 20,
            "compression_window_turns": 15,
            "compress_every_n_turns": 10,
            "llm_perception_mode": "lite",
        },
        "128k": {
            "model_context_window": 128000,
            "context_budget": {
                "system_prompt": 3000,
                "cognitive_fingerprint_ratio": 0.03,
                "topic_timelines_ratio": 0.05,
                "compressed_summaries_ratio": 0.08,
                "injected_memories_ratio": 0.15,
                "recent_turns_ratio": 0.38,
                "response_reserve_ratio": 0.15,
                "buffer_ratio": 0.11,
            },
            "retrieval_top_k": 15,
            "working_memory_recent_turns": 40,
            "compression_window_turns": 30,
            "compress_every_n_turns": 25,
            "llm_perception_mode": "full",
        },
        "256k": {
            "model_context_window": 256000,
            "context_budget": {
                "system_prompt": 4000,
                "cognitive_fingerprint_ratio": 0.02,
                "topic_timelines_ratio": 0.03,
                "compressed_summaries_ratio": 0.05,
                "injected_memories_ratio": 0.15,
                "recent_turns_ratio": 0.35,
                "response_reserve_ratio": 0.15,
                "buffer_ratio": 0.05,
            },
            "retrieval_top_k": 30,
            "working_memory_recent_turns": 80,
            "compression_window_turns": 50,
            "compress_every_n_turns": 50,
            "llm_perception_mode": "full",
        },
        "1M": {
            "model_context_window": 1048576,
            "context_budget": {
                "system_prompt": 4000,
                "cognitive_fingerprint_ratio": 0.005,
                "topic_timelines_ratio": 0.01,
                "compressed_summaries_ratio": 0.02,
                "injected_memories_ratio": 0.20,
                "recent_turns_ratio": 0.35,
                "response_reserve_ratio": 0.15,
                "buffer_ratio": 0.03,
            },
            "retrieval_top_k": 80,
            "working_memory_recent_turns": 200,
            "compression_window_turns": 100,
            "compress_every_n_turns": 100,
            "llm_perception_mode": "full",
        },
    }

    @classmethod
    def get(cls) -> Dict[str, Any]:
        """Return merged configuration (file + env + defaults), cached."""
        if cls._data is None:
            with cls._lock:
                if cls._data is None:
                    cls._data = cls._load()
        return cls._data

    @classmethod
    def _load(cls) -> Dict[str, Any]:
        """Load config.json, overlay env vars, fill missing with defaults."""
        config_path = Path(__file__).parent.parent / "config.json"
        file_data: Dict[str, Any] = {}
        if config_path.is_file():
            try:
                with open(config_path, encoding="utf-8") as fh:
                    file_data = json.load(fh)
            except (IOError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Failed to load config from {config_path}: {exc}") from exc

        # Start from defaults and overlay file data (user wins over default)
        merged: Dict[str, Any] = {**cls.DEFAULTS}
        merged.update(file_data)

        # Ensure context_tiers is always present (from code, never from file in v4)
        merged.setdefault("context_tiers", cls.CONTEXT_TIERS)

        # Override sensitive config from environment variables
        merged["gateway_token"] = os.environ.get(
            "HMS_GATEWAY_TOKEN",
            merged.get("gateway_token", ""),
        )
        merged["gateway_url"] = os.environ.get(
            "HMS_GATEWAY_URL",
            merged.get("gateway_url", "http://127.0.0.1:18789"),
        )

        return merged

    @classmethod
    def reload(cls) -> None:
        """Force a fresh load from disk."""
        with cls._lock:
            cls._data = cls._load()
