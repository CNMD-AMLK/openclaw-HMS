"""
HMS v2 — Forgetting Engine.

Ebbinghaus-inspired multi-factor memory decay with emotional modulation.
Optimized from v1 with better half-life scaling and immortal guards.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ForgettingEngine:
    """
    Manages memory decay, strength evaluation, and forgetting decisions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self._cache_path = self.cfg.get("decay_cache_path", "cache/decay_state.json")
        self._states: Dict[str, Dict[str, Any]] = {}
        self._base_threshold = self.cfg.get("forget_base_threshold", 0.15)
        self._emotion_slowdown = self.cfg.get("emotion_decay_slowdown_factor", 2.0)
        self._reinforce_delta = self.cfg.get("reinforcement_confidence_delta", 0.05)
        self.load_decay_state()

    # ==================================================================
    # Persistence
    # ==================================================================

    def load_decay_state(self) -> None:
        if os.path.isfile(self._cache_path):
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    self._states = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._states = {}
        else:
            self._states = {}

    def save_decay_state(self) -> None:
        os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(self._states, f, ensure_ascii=False, indent=2)

    # ==================================================================
    # Access / reinforcement hooks
    # ==================================================================

    def update_on_access(self, memory_id: str) -> None:
        """Call on every memory_recall hit."""
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
        self.save_decay_state()

    def update_on_reinforce(self, memory_id: str) -> None:
        """Call on every collision reinforcement."""
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
        self.save_decay_state()

    # ==================================================================
    # Strength calculation
    # ==================================================================

    def calculate_strength(
        self,
        memory_id: str,
        current_time: Optional[datetime] = None,
    ) -> float:
        """
        Multi-factor memory strength:

        S = importance
            * (1 + arousal * emotion_slowdown)
            * (1 + reinforced * reinforce_weight)
            * (1 + confidence * 0.5)
            * (1 + related * 0.05)
            * time_decay

        time_decay = e^(-0.693 * hours / half_life)
        half_life scales with importance: 168h * (importance / 5)
        """
        s = self._states.get(memory_id)
        if not s or not s.get("last_accessed"):
            return 0.0

        now = current_time or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        last = datetime.fromisoformat(s["last_accessed"])
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)

        hours = max(0.0, (now - last).total_seconds() / 3600.0)

        importance = s.get("importance", 5.0)
        arousal = s.get("emotional_arousal", 0.0)
        reinforced = s.get("times_reinforced", 0)
        confidence = s.get("belief_confidence", 0.5)
        related = s.get("related_count", 0)

        # Half-life scales with importance (24h–720h)
        half_life = 168.0 * (importance / 5.0)
        half_life = max(24.0, min(720.0, half_life))

        time_decay = math.exp(-0.693147 * hours / half_life)

        emo_mult = 1.0 + arousal * self._emotion_slowdown
        rein_mult = 1.0 + reinforced * 0.15
        conf_mult = 1.0 + confidence * 0.5
        rel_mult = 1.0 + related * 0.05

        strength = importance * emo_mult * rein_mult * conf_mult * rel_mult * time_decay
        return round(max(0.0, strength), 4)

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
        """Memories that should never be forgotten."""
        importance = mem.get("importance", 0)
        if importance >= 9:
            return True
        meta = mem.get("metadata") or {}
        if meta.get("belief_strength") == "certain" and meta.get("belief_confidence", 0) >= 0.95:
            return True
        # Cognitive fingerprint entries
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
            except Exception:
                pass
        self.save_decay_state()
        return deleted

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _sync_from_memory(self, mem: Dict[str, Any]) -> None:
        """Pull metadata into decay state if not already tracked."""
        mid = mem.get("id", "")
        if not mid:
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
            except Exception:
                return {}
        return {}


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
