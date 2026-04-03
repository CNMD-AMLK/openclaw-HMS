"""
HMS v3.6 — Common utilities.

Shared functions used across multiple modules.
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Lazy-load jieba to avoid hard dependency
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
    """
    Tokenize text with Chinese-aware segmentation.

    Uses jieba if available, falls back to char bigram for Chinese
    and whitespace split for ASCII-dominant text.
    """
    if not text:
        return []

    cn_ratio = sum(1 for c in text if "\u4e00" <= c <= "\u9fff") / max(len(text), 1)

    if cn_ratio > 0.1:
        jieba = _get_jieba()
        if jieba is not None:
            return [w for w in jieba.lcut(text) if w.strip()]

    # Fallback: char bigram for mixed/CJK text
    if cn_ratio > 0.05:
        return _char_bigrams(text)

    # ASCII-dominant: simple whitespace split
    return text.lower().split()


def _char_bigrams(text: str) -> List[str]:
    """Split text into character bigrams for CJK fallback tokenization."""
    cleaned = re.sub(r"\s+", " ", text.lower().strip())
    if len(cleaned) < 2:
        return [cleaned]
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation based on character types:
    - Chinese/Japanese/Korean: ~1.5 chars/token
    - English/ASCII: ~4 chars/token
    - Code/special: ~3 chars/token

    This is an approximation. For accurate counts, use tiktoken or similar.
    """
    if not text:
        return 0
    cn = sum(
        1
        for c in text
        if "\u4e00" <= c <= "\u9fff"
        or "\u3040" <= c <= "\u309f"
        or "\u30a0" <= c <= "\u30ff"
    )
    ascii_chars = sum(1 for c in text if c.isascii() and not c.isspace())
    other = len(text) - cn - ascii_chars
    return int(cn / 1.5 + ascii_chars / 4 + other / 3)


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    if not text or len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def sanitize_text(text: str) -> str:
    """Remove control characters (preserving \\n\\t), replace null bytes, clean invisible chars."""
    if not isinstance(text, str):
        return text
    # Replace null bytes
    text = text.replace("\x00", "")
    text = text.replace("\u0000", "")
    # Remove other control characters (preserve \n \t \r)
    import re
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Remove BOM
    text = text.lstrip("\ufeff")
    return text
