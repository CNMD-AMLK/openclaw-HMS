"""
HMS v4 — Typed exception hierarchy.

Inspired by Rust's thiserror/anyhow patterns:
  - Each error variant carries rich context (like Rust enum with named fields)
  - `is_retryable()` method for auto-retry decisions (like Rust `is_retryable()`)
  - `from_*` constructor factories (like Rust `From<T>` trait)
  - Clear inheritance tree for precise `except` matching

Usage:
    try:
        adapter.store("test", "fact", 5, "{}")
    except StoreError as e:
        if e.is_retryable():
            retry_later(...)
        elif e.is_transient():
            log_and_continue(...)
        else:
            raise
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ============================================================
# Base HMS error
# ============================================================


class HMSError(Exception):
    """Base class for all HMS exceptions.

    Carries structured context (like Rust error enum payloads)
    and an optional retryable flag for automatic retry decisions.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "",
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self._retryable = retryable
        self.__cause__ = cause

    def is_retryable(self) -> bool:
        """Whether this error should trigger an automatic retry.

        Mirrors Rust ApiError::is_retryable().
        """
        return self._retryable

    def is_transient(self) -> bool:
        """Temporary failure that may resolve without intervention.

        Network blips, rate limits, etc. Retry after backoff.
        """
        return self._retryable or self.code in (
            "rate_limited",
            "connection_error",
            "timeout",
            "server_error",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize error context for logging/telemetry."""
        return {
            "type": type(self).__name__,
            "code": self.code,
            "message": str(self),
            "retryable": self._retryable,
            "details": self.details,
        }

    @classmethod
    def from_exception(
        cls, exc: Exception, *, context: Optional[Dict[str, Any]] = None
    ) -> "HMSError":
        """Factory: wrap any exception into an HMS error.

        Like Rust `From<T>` trait for automatic error conversion.
        """
        details = dict(context or {})
        details["original_error"] = str(exc)
        return cls(
            message=f"{type(exc).__name__}: {exc}",
            code=type(exc).__name__.lower(),
            details=details,
            cause=exc,
        )


# ============================================================
# Storage errors
# ============================================================


class StoreError(HMSError):
    """Memory store operation failed.

    Variants (identified by `code`):
      - dedup_failed: similarity check error
      - api_unavailable: Gateway/tool unreachable
      - invalid_input: malformed text/metadata
      - storage_full: underlying store capacity exceeded
    """

    @classmethod
    def from_gateway_error(
        cls, status_code: int, body: str = "", endpoint: str = ""
    ) -> "StoreError":
        """Create from an HTTP error response (Like Rust From<reqwest::Error>)."""
        retryable = status_code in (429, 500, 502, 503, 504)
        return cls(
            message=f"Gateway store error: HTTP {status_code}",
            code="api_unavailable",
            details={"status_code": status_code, "body": body[:500], "endpoint": endpoint},
            retryable=retryable,
        )

    @classmethod
    def from_dedup_error(cls, similarity: float, threshold: float, matched_id: str = "") -> "StoreError":
        retryable = False
        return cls(
            message=f"Dedup conflict: similarity={similarity:.3f} >= threshold={threshold:.3f}",
            code="dedup_failed",
            details={"similarity": similarity, "threshold": threshold, "matched_id": matched_id},
            retryable=retryable,
        )


class RecallError(HMSError):
    """Memory recall operation failed.

    Variants:
      - api_unavailable: Gateway/tool unreachable
      - query_error: malformed query
      - decode_error: invalid response format
    """

    @classmethod
    def from_gateway_error(
        cls, status_code: int, body: str = "", endpoint: str = ""
    ) -> "RecallError":
        retryable = status_code in (429, 500, 502, 503, 504)
        return cls(
            message=f"Gateway recall error: HTTP {status_code}",
            code="api_unavailable",
            details={"status_code": status_code, "body": body[:500], "endpoint": endpoint},
            retryable=retryable,
        )


# ============================================================
# Context errors
# ============================================================


class ContextError(HMSError):
    """Context composition or management failed.

    Variants:
      - budget_exceeded: token budget exceeded minimum floor
      - corruption: corrupted state file
      - serialization: JSON encode/decode failure
    """


# ============================================================
# Perception errors
# ============================================================


class PerceptionError(HMSError):
    """Perception analysis failed.

    Variants:
      - llm_unavailable: no LLM response and heuristics disabled
      - parse_error: invalid LLM JSON response
    """


# ============================================================
# Consolidation errors
# ============================================================


class ConsolidationError(HMSError):
    """Consolidation pipeline failed.

    Variants:
      - partial_failure: some steps succeeded, others failed
      - full_failure: pipeline completely failed
    """
