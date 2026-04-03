"""HMS v4 — Scripts package.

All engine modules are exposed here for convenience.
v4 additions: reconstructive_recall, dream_engine, creative_assoc.
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Core engines
from .models import (  # noqa: F401
    MemoryBelief, CognitiveFingerprint, TopicTimeline,
    DecayState, MemoryMeta, EmotionalTrace, CompressedSummary,
    MemoryType, MemorySource, BeliefStrength,
)
from .config_loader import Config  # noqa: F401
from .memory_manager import MemoryManager, MemoryAdapter, main  # noqa: F401
from .perception import PerceptionEngine  # noqa: F401
from .collision import CollisionEngine  # noqa: F401
from .context_manager import ContextManager  # noqa: F401
from .consolidation import ConsolidationEngine  # noqa: F401
from .forgetting import ForgettingEngine, MemoryOverwriter  # noqa: F401
from .llm_analyzer import LLMAnalyzer  # noqa: F401
from .embed_cache import EmbeddingCache, prefilter_for_collision  # noqa: F401

# v4 new modules
from .reconstructive_recall import ReconstructiveRecaller  # noqa: F401
from .dream_engine import DreamEngine  # noqa: F401
from .creative_assoc import CreativeAssociator  # noqa: F401

# Utilities
from .utils import estimate_tokens, tokenize  # noqa: F401
from .file_utils import file_lock, atomic_write_json, safe_read_json  # noqa: F401


def setup_logging(level: int = __import__('logging').INFO) -> None:
    """Configure basic logging for HMS."""
    import logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(handler)
