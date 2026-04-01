"""
HMS v2 — Common utilities.

Shared functions used across multiple modules.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation: Chinese ~1.5 char/token, English ~4 char/token.
    
    This is an approximation. For accurate counts, use tiktoken or similar.
    """
    if not text:
        return 0
    cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    en = len(text) - cn
    return int(cn / 1.5 + en / 4)


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator
