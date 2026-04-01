"""
HMS v3 — Common utilities.

Shared functions used across multiple modules.
"""

from __future__ import annotations


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
    cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u309f" or "\u30a0" <= c <= "\u30ff")
    ascii_chars = sum(1 for c in text if c.isascii() and not c.isspace())
    other = len(text) - cn - ascii_chars
    return int(cn / 1.5 + ascii_chars / 4 + other / 3)


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
