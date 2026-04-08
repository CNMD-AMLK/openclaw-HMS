"""
HMS v5 — Text utilities (adapted from hms-v4 scripts/utils.py).
"""

from __future__ import annotations
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

_jieba_available = False
_jieba = None


def _get_jieba():
    global _jieba, _jieba_available
    if _jieba is None:
        try:
            import jieba as _j
            _jieba = _j
            _jieba_available = True
        except ImportError:
            _jieba_available = False
    return _jieba


def tokenize(text: str) -> List[str]:
    """Tokenize text with Chinese-aware segmentation."""
    if not text:
        return []
    cn_ratio = sum(1 for c in text if "\u4e00" <= c <= "\u9fff") / max(len(text), 1)
    if cn_ratio > 0.1:
        jieba = _get_jieba()
        if jieba is not None:
            return [w for w in jieba.lcut(text) if w.strip()]
    if cn_ratio > 0.05:
        return _char_bigrams(text)
    return text.lower().split()


def _char_bigrams(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text.lower().strip())
    if len(cleaned) < 2:
        return [cleaned]
    return [cleaned[i:i + 2] for i in range(len(cleaned) - 1)]


def estimate_tokens(text: str) -> int:
    """Rough token estimation."""
    if not text:
        return 0
    cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u309f" or "\u30a0" <= c <= "\u30ff")
    ascii_chars = sum(1 for c in text if c.isascii() and not c.isspace())
    other = len(text) - cn - ascii_chars
    return int(cn / 1.5 + ascii_chars / 4 + other / 3)


def sanitize_text(text: str) -> str:
    """Remove control characters."""
    if not isinstance(text, str):
        return text
    text = text.replace("\x00", "").replace("\u0000", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = text.lstrip("\ufeff")
    return text
