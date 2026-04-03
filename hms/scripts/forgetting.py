"""
HMS — Forgetting Engine.

Ebbinghaus-inspired multi-factor memory decay with emotional modulation.
Optimized from v1 with better half-life scaling and immortal guards.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .file_utils import atomic_write_json, safe_read_json, file_lock
from .models import DecayState

logger = logging.getLogger(__name__)


class ForgettingEngine:
    """
    Manages memory decay, strength evaluation, and forgetting decisions.
    Uses DecayState from models.py for strength calculation to avoid duplication.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self._cache_path = self.cfg.get("decay_cache_path", "cache/decay_state.json")
        self._states: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self._base_threshold = self.cfg.get("forget_base_threshold", 0.15)
        self._emotion_slowdown = self.cfg.get("emotion_decay_slowdown_factor", 2.0)
        self._reinforce_delta = self.cfg.get("reinforcement_confidence_delta", 0.05)
        self.load_decay_state()

    # ==================================================================
    # Persistence
    # ==================================================================

    def load_decay_state(self) -> None:
        self._states = safe_read_json(self._cache_path, {})

    def save_decay_state(self) -> None:
        with file_lock(self._cache_path):
            atomic_write_json(self._cache_path, self._states)

    # ==================================================================
    # Access / reinforcement hooks
    # ==================================================================

    def update_on_access(self, memory_id: str) -> None:
        """Call on every memory_recall hit.

        Only marks the state dirty; caller must flush() periodically.
        """
        now = datetime.now(timezone.utc).isoformat()
        s = self._states.get(memory_id)
        if s:
            s["access_count"] = s.get("access_count", 0) + 1
            s["last_accessed"] = now
        else:
            self._states[memory_id] = {
                "memory_id": memory_id,
                "last_accessed": now,
                "access_count": 1,
                "times_reinforced": 0,
                "importance": 5.0,
                "emotional_arousal": 0.0,
                "emotional_valence": 0.0,
                "belief_confidence": 0.5,
                "related_count": 0,
            }
        self._dirty = True

    def update_on_reinforce(self, memory_id: str) -> None:
        """Call on every collision reinforcement.

        Only marks the state dirty; caller must flush() periodically.
        """
        s = self._states.get(memory_id)
        if s:
            s["times_reinforced"] = s.get("times_reinforced", 0) + 1
        else:
            self._states[memory_id] = {
                "memory_id": memory_id,
                "last_accessed": datetime.now(timezone.utc).isoformat(),
                "access_count": 0,
                "times_reinforced": 1,
                "importance": 5.0,
                "emotional_arousal": 0.0,
                "emotional_valence": 0.0,
                "belief_confidence": 0.5,
                "related_count": 0,
            }
        self._dirty = True

    def flush(self) -> None:
        """Persist decay state to disk if dirty."""
        if self._dirty:
            import tempfile as _tmp
            _dir = os.path.dirname(self._cache_path) or "."
            os.makedirs(_dir, exist_ok=True)
            _fd, _tmp_path = _tmp.mkstemp(dir=_dir, suffix=".tmp")
            try:
                with os.fdopen(_fd, "w", encoding="utf-8") as _f:
                    json.dump(self._states, _f, ensure_ascii=False, indent=2)
                os.replace(_tmp_path, self._cache_path)
            except Exception:
                try:
                    os.unlink(_tmp_path)
                except OSError:
                    pass
                raise
            self._dirty = False

    # ==================================================================
    # Strength calculation (uses DecayState from models.py)
    # ==================================================================

    def calculate_strength(
        self,
        memory_id: str,
        current_time: Optional[datetime] = None,
    ) -> float:
        """
        Multi-factor memory strength calculation.
        Delegates to DecayState.calculate_strength() to avoid code duplication.
        """
        s = self._states.get(memory_id)
        if not s or not s.get("last_accessed"):
            return 0.0

        # Create a DecayState instance from the stored state
        decay_state = DecayState(
            memory_id=memory_id,
            last_accessed=s.get("last_accessed", ""),
            access_count=s.get("access_count", 0),
            times_reinforced=s.get("times_reinforced", 0),
            importance=s.get("importance", 5.0),
            emotional_arousal=s.get("emotional_arousal", 0.0),
            emotional_valence=s.get("emotional_valence", 0.0),
            belief_confidence=s.get("belief_confidence", 0.5),
            related_count=s.get("related_count", 0),
        )

        now = current_time or datetime.now(timezone.utc)
        return decay_state.calculate_strength(
            now,
            emotion_slowdown=self._emotion_slowdown,
            reinforcement_weight=0.15,
            half_life_hours=168.0,
        )

    # ==================================================================
    # Dynamic threshold
    # ==================================================================

    def get_threshold(self, memory_id: str) -> float:
        """Dynamic forget threshold = base + importance bonus + type bonus."""
        s = self._states.get(memory_id, {})
        importance = s.get("importance", 5.0)
        imp_bonus = (importance / 10.0) * 0.3
        mem_type = s.get("memory_type", "episodic")
        type_bonus = {
            "procedural": 0.2,
            "semantic": 0.1,
            "episodic": 0.0,
            "prospective": 0.15,
        }.get(mem_type, 0.0)
        return round(self._base_threshold + imp_bonus + type_bonus, 4)

    # ==================================================================
    # Immortal guard
    # ==================================================================

    @staticmethod
    def _is_immortal(mem: Dict[str, Any]) -> bool:
        """Memories that should never be forgotten.

        Requires a minimum evidence threshold to avoid single-point memories
        from becoming immortal due to overconfident scoring.
        """
        importance = mem.get("importance", 0)
        if importance >= 9:
            return True
        meta = mem.get("metadata") or {}
        belief_confidence = meta.get("belief_confidence", 0)
        belief_strength = meta.get("belief_strength", "")
        evidence_count = len(meta.get("derived_from", [])) or len(meta.get("evidence", []))
        if belief_strength == "certain" and belief_confidence >= 0.95 and evidence_count >= 3:
            return True
        if meta.get("source") == "consolidated" and meta.get("memory_type") == "semantic":
            return True
        return False

    # ==================================================================
    # Evaluate all
    # ==================================================================

    def evaluate_all(
        self,
        lancedb_memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate every memory against its strength and threshold."""
        to_forget: List[str] = []
        to_keep: List[str] = []
        strengths: Dict[str, float] = {}
        now = datetime.now(timezone.utc)

        for mem in lancedb_memories:
            mid = mem.get("id", "")
            if not mid:
                logger.warning("Skipping memory with empty ID in evaluate_all")
                continue

            self._sync_from_memory(mem)
            strength = self.calculate_strength(mid, now)
            threshold = self.get_threshold(mid)
            strengths[mid] = strength

            if self._is_immortal(mem):
                to_keep.append(mid)
                continue

            if strength < threshold:
                to_forget.append(mid)
            else:
                to_keep.append(mid)

        total = len(lancedb_memories)
        return {
            "to_forget": to_forget,
            "to_keep": to_keep,
            "report": {
                "total_evaluated": total,
                "to_forget_count": len(to_forget),
                "to_keep_count": len(to_keep),
                "forget_ratio": round(len(to_forget) / total, 3) if total else 0,
                "avg_strength": round(
                    sum(strengths.values()) / len(strengths), 3
                ) if strengths else 0,
                "strengths": strengths,
            },
        }

    def execute_forgetting(
        self,
        to_forget_ids: List[str],
        memory_forget_func: Any,
    ) -> int:
        """Delete forgotten memories via injected memory_forget."""
        deleted = 0
        for mid in to_forget_ids:
            try:
                memory_forget_func(mid)
                self._states.pop(mid, None)
                deleted += 1
            except Exception as e:
                logger.debug("Failed to forget memory %s: %s", mid, e)
        self._dirty = True
        self.flush()
        return deleted

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _sync_from_memory(self, mem: Dict[str, Any]) -> None:
        """Pull metadata into decay state if not already tracked."""
        mid = mem.get("id", "")
        if not mid:
            logger.warning("Skipping memory with empty ID in _sync_from_memory")
            return
        meta = self._parse_meta(mem.get("metadata"))
        s = self._states.get(mid, {})
        if mid not in self._states:
            self._states[mid] = {
                "memory_id": mid,
                "last_accessed": mem.get("created_at", datetime.now(timezone.utc).isoformat()),
                "access_count": 0,
                "times_reinforced": 0,
                "importance": float(mem.get("importance", 5)),
                "emotional_arousal": meta.get("emotional_arousal", 0.0),
                "emotional_valence": meta.get("emotional_valence", 0.0),
                "belief_confidence": meta.get("belief_confidence", 0.5),
                "related_count": len(meta.get("related_memory_ids", [])),
                "memory_type": meta.get("memory_type", "episodic"),
            }

    @staticmethod
    def _parse_meta(meta: Any) -> Dict[str, Any]:
        if isinstance(meta, dict):
            return meta
        if isinstance(meta, str):
            try:
                return json.loads(meta)
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}

    def sync_consistency(
        self,
        memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Reconcile decay_state with actual memory store.
        Returns a report of fixes applied.
        """
        report = {"synced": 0, "orphaned_removed": 0, "missing_added": 0}
        memory_ids = set()

        for mem in memories:
            mid = mem.get("id", "")
            if not mid:
                continue
            memory_ids.add(mid)
            if mid not in self._states:
                self._sync_from_memory(mem)
                report["missing_added"] += 1
            else:
                # Update importance from memory store
                mem_imp = float(mem.get("importance", 5))
                state_imp = self._states[mid].get("importance", 5)
                if mem_imp != state_imp:
                    self._states[mid]["importance"] = mem_imp
                    report["synced"] += 1

        # Remove orphaned states (memory was deleted but decay state remains)
        orphaned = [mid for mid in self._states if mid not in memory_ids]
        for mid in orphaned:
            del self._states[mid]
            report["orphaned_removed"] += 1

        if report["synced"] or report["orphaned_removed"] or report["missing_added"]:
            self.save_decay_state()

        return report


# ======================================================================
# Self-test
# ======================================================================


def _self_test():
    """Run: python -m hms.scripts.forgetting"""
    import tempfile

    path = tempfile.mktemp(suffix=".json")
    try:
        fe = ForgettingEngine({"decay_cache_path": path})

        # Test access
        fe.update_on_access("m1")
        assert "m1" in fe._states

        # Test strength calculation
        strength = fe.calculate_strength("m1")
        assert strength > 0
        print(f"[strength] m1 = {strength}")

        # Test threshold
        th = fe.get_threshold("m1")
        assert th > 0
        print(f"[threshold] m1 = {th}")

        # Test immortal
        immortal_mem = {"id": "vip", "importance": 10, "metadata": {}}
        assert fe._is_immortal(immortal_mem)
        normal_mem = {"id": "normal", "importance": 5, "metadata": {}}
        assert not fe._is_immortal(normal_mem)
        print("[immortal] OK")

        print("✓ All self-tests passed.")
    finally:
        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    _self_test()
