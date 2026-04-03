"""
HMS v4 — Forgetting Engine (enhanced with MemoryOverwriter).

Ebbinghaus-inspired multi-factor memory decay with emotional modulation.
v4 adds: MemoryOverwriter for conflict resolution and belief supersession.
"""

from __future__ import annotations

import json
import logging
import os
import time
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
    # Evaluate all (v4: calls MemoryOverwriter for conflicts)
    # ==================================================================

    def evaluate_all(
        self,
        lancedb_memories: List[Dict[str, Any]],
        overwriter: Optional["MemoryOverwriter"] = None,
    ) -> Dict[str, Any]:
        """Evaluate every memory against its strength and threshold.

        v4: if an overwriter is provided, check for conflicting memories
        and handle them before the standard forget/keep decision.
        """
        to_forget: List[str] = []
        to_keep: List[str] = []
        to_supersede: List[Dict[str, Any]] = []
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

            # v4: check for conflicts via overwriter
            if overwriter is not None:
                conflict_result = overwriter.check_and_handle(mem, lancedb_memories)
                if conflict_result.get("superseded"):
                    to_supersede.append(conflict_result)
                    continue  # skip normal forget/keep for superseded

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
            "superseded": to_supersede,
            "report": {
                "total_evaluated": total,
                "to_forget_count": len(to_forget),
                "to_keep_count": len(to_keep),
                "superseded_count": len(to_supersede),
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
                mem_imp = float(mem.get("importance", 5))
                state_imp = self._states[mid].get("importance", 5)
                if mem_imp != state_imp:
                    self._states[mid]["importance"] = mem_imp
                    report["synced"] += 1

        orphaned = [mid for mid in self._states if mid not in memory_ids]
        for mid in orphaned:
            del self._states[mid]
            report["orphaned_removed"] += 1

        if report["synced"] or report["orphaned_removed"] or report["missing_added"]:
            self.save_decay_state()

        return report


# ======================================================================
# MemoryOverwriter — v4: Conflict resolution and belief supersession
# ======================================================================

class MemoryOverwriter:
    """
    Handles memory conflicts by superseding rather than deleting old beliefs.

    When new evidence contradicts an existing belief:
      1. Mark the old belief as 'superseded' (not deleted)
      2. Downgrade its confidence
      3. Add a superseded_by reference pointing to the new evidence
      4. Keep it retrievable for historical context
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self._conflict_threshold = self.cfg.get("collision_threshold", 0.7)
        self._downgrade_factor = self.cfg.get("overwriting_downgrade_factor", 0.3)
        logger.debug("MemoryOverwriter initialized (downgrade=%.2f)", self._downgrade_factor)

    def handle_conflict(
        self,
        old_belief: Dict[str, Any],
        new_evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle a conflict between an old belief and new evidence.

        Args:
            old_belief: existing memory entry to potentially supersede
            new_evidence: incoming memory/evidence that contradicts

        Returns:
            Resolution result dict with action taken and updated states.
        """
        if not old_belief or not new_evidence:
            return {"action": "no_conflict", "reason": "empty input"}

        old_text = old_belief.get("text", "").lower()
        new_text = new_evidence.get("text", "").lower()

        if not old_text or not new_text:
            return {"action": "no_conflict", "reason": "no text content"}

        # Check if this is actually a conflict (not just different topics)
        is_conflict = self._detect_conflict(old_text, new_text)

        if not is_conflict:
            return {"action": "no_conflict", "reason": "texts not conflicting"}

        old_meta = old_belief.get("metadata", {})
        old_confidence = old_meta.get("belief_confidence", 0.5) if isinstance(old_meta, dict) else 0.5
        new_confidence = new_evidence.get("metadata", {}).get("belief_confidence", 0.5)

        # If new evidence has higher confidence, supersede the old
        if new_confidence > old_confidence:
            result = self._supersede(old_belief, new_evidence)
        else:
            # Keep old, just downgrade new's importance suggestion
            result = {
                "action": "keep_old",
                "old_id": old_belief.get("id", ""),
                "new_confidence_downgraded": round(new_confidence * 0.8, 3),
                "reason": "old belief has higher confidence",
            }

        return result

    def check_and_handle(
        self,
        candidate: Dict[str, Any],
        all_memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Scan all memories for potential conflicts with the candidate.

        Called during evaluate_all to handle conflicts before forget/keep.
        """
        candidate_text = candidate.get("text", "").lower()
        if not candidate_text:
            return {"superseded": False}

        # Check if candidate is already superseded
        meta = candidate.get("metadata", {})
        if isinstance(meta, dict) and meta.get("superseded"):
            return {"superseded": True, "reason": "already superseded"}

        # Actually detect conflicts with other memories
        for other in all_memories:
            if other.get("id", "") == candidate.get("id", ""):
                continue

            other_text = other.get("text", "").lower()
            if not other_text:
                continue

            # FIX: actually call _detect_conflict!
            if self._detect_conflict(candidate_text, other_text):
                # Attempt to handle the conflict (supersede older if newer is more confident)
                return self.handle_conflict(other, candidate)

    def _detect_conflict(self, text_a: str, text_b: str) -> bool:
        """
        Heuristic conflict detection.

        Two texts conflict if:
        - They share keywords but have opposing sentiment/statement
        - They mention the same entity with different attributes
        """
        from hms.scripts.utils import tokenize

        tokens_a = set(tokenize(text_a))
        tokens_b = set(tokenize(text_b))

        # Need some overlap to be considered related
        if not tokens_a or not tokens_b:
            return False

        overlap = tokens_a & tokens_b
        overlap_ratio = len(overlap) / max(len(tokens_a), len(tokens_b), 1)

        if overlap_ratio < 0.3:
            return False  # Not related enough to conflict

        # Check for negation/opposition patterns
        negation_words = {"不", "没", "非", "否", "anti", "not", "never", "neither", "don't", "doesn't", "not "}
        has_neg_a = any(w in text_a for w in negation_words)
        has_neg_b = any(w in text_b for w in negation_words)

        # XOR: one has negation, the other doesn't → likely conflict
        if has_neg_a != has_neg_b and overlap_ratio > 0.4:
            return True

        # Opposite sentiment words
        opposite = {
            "好": "坏", "喜欢": "讨厌", "爱": "恨", "高": "低",
            "hot": "cold", "good": "bad", "happy": "sad", "like": "dislike",
        }
        for pos, neg in opposite.items():
            a_has_pos = pos in text_a
            b_has_neg = neg in text_b
            if a_has_pos and b_has_neg and overlap_ratio > 0.2:
                return True
            a_has_neg = neg in text_a
            b_has_pos = pos in text_b
            if a_has_neg and b_has_pos and overlap_ratio > 0.2:
                return True

        return False

    def _supersede(
        self,
        old: Dict[str, Any],
        new: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Supersede old belief with new evidence.

        Marks old as 'superseded' instead of deleting it.
        """
        old_meta = old.get("metadata", {})
        if isinstance(old_meta, str):
            try:
                old_meta = json.loads(old_meta)
            except (json.JSONDecodeError, ValueError):
                old_meta = {}
        if not isinstance(old_meta, dict):
            old_meta = {}

        # Mark as superseded
        old_meta["superseded"] = True
        old_meta["superseded_at"] = datetime.now(timezone.utc).isoformat()
        old_meta["superseded_by"] = new.get("id", "unknown")
        old_meta["superseded_original_confidence"] = old_meta.get("belief_confidence", 0.5)

        # Downgrade confidence
        old_meta["belief_confidence"] = round(
            old_meta.get("belief_confidence", 0.5) * self._downgrade_factor,
            3,
        )

        result = {
            "action": "superseded",
            "old_id": old.get("id", ""),
            "new_id": new.get("id", ""),
            "updated_metadata": old_meta,
            "old_confidence": self._downgrade_factor,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Superseded memory %s (confidence downgraded to %.3f)",
            old.get("id", ""),
            old_meta["belief_confidence"],
        )

        return result

    def _downgrade_confidence(
        self,
        belief: Dict[str, Any],
        factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Downgrade a belief's confidence without deleting it.

        Used for memories that are partially contradicted
        but not fully superseded.
        """
        f = factor if factor is not None else self._downgrade_factor
        meta = belief.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, ValueError):
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        original_conf = meta.get("belief_confidence", 0.5)
        meta["belief_confidence"] = round(original_conf * f, 3)
        meta["confidence_downgraded_at"] = datetime.now(timezone.utc).isoformat()
        meta["confidence_downgrade_reason"] = "partially contradicted"

        return {
            "action": "downgraded",
            "memory_id": belief.get("id", ""),
            "original_confidence": original_conf,
            "new_confidence": meta["belief_confidence"],
            "metadata": meta,
        }


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

        # ─── v4: MemoryOverwriter tests ───
        ow = MemoryOverwriter()

        # No conflict case
        no_conflict = ow.handle_conflict(
            {"text": "我喜欢 Python", "id": "a", "metadata": {"belief_confidence": 0.8}},
            {"text": "今天天气不错", "id": "b", "metadata": {"belief_confidence": 0.9}},
        )
        assert no_conflict["action"] == "no_conflict"
        print("[overwriter] non-conflict OK")

        # Conflict case with supersession
        supersede = ow.handle_conflict(
            {"text": "我最喜欢 Python", "id": "old1", "metadata": {"belief_confidence": 0.5}},
            {"text": "我现在最喜欢 Rust 不是 Python", "id": "new1", "metadata": {"belief_confidence": 0.9}},
        )
        assert supersede["action"] in ("superseded", "keep_old")
        print(f"[overwriter] conflict handling OK (action={supersede['action']})")

        # Downgrade test
        old_belief = {"id": "down1", "metadata": {"belief_confidence": 0.8}}
        downgrade_result = ow._downgrade_confidence(old_belief, factor=0.3)
        assert downgrade_result["new_confidence"] == round(0.8 * 0.3, 3)
        print(f"[overwriter] downgrade OK (0.8 → {downgrade_result['new_confidence']})")

        print("✓ All self-tests passed (v4 with MemoryOverwriter).")
    finally:
        if os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    _self_test()
