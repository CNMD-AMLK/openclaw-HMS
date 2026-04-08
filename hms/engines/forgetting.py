"""
HMS v5 — Forgetting Engine + MemoryOverwriter (adapted from hms-v4 scripts/forgetting.py).
"""

from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hms.utils.file_utils import atomic_write_json, safe_read_json, file_lock
from hms.core.models import DecayState

logger = logging.getLogger("hms.forgetting")


class ForgettingEngine:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self._cache_path = config.get("decay_cache_path", os.path.join(config.get("dataDir", "data"), "decay_state.json"))
        self._states: Dict[str, Dict] = {}
        self._dirty = False
        self._base_threshold = config.get("forget_base_threshold", 0.15)
        self._emotion_slowdown = config.get("emotion_decay_slowdown_factor", 2.0)
        self.load_decay_state()

    def load_decay_state(self) -> None:
        self._states = safe_read_json(self._cache_path, {})

    def save_decay_state(self) -> None:
        with file_lock(self._cache_path):
            atomic_write_json(self._cache_path, self._states)
        self._dirty = False

    def update_on_access(self, memory_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        s = self._states.get(memory_id)
        if s:
            s["access_count"] = s.get("access_count", 0) + 1
            s["last_accessed"] = now
        else:
            self._states[memory_id] = {
                "memory_id": memory_id, "last_accessed": now,
                "access_count": 1, "times_reinforced": 0,
                "importance": 5.0, "emotional_arousal": 0.0,
                "emotional_valence": 0.0, "belief_confidence": 0.5,
                "related_count": 0,
            }
        self._dirty = True

    def update_on_reinforce(self, memory_id: str) -> None:
        s = self._states.get(memory_id)
        if s:
            s["times_reinforced"] = s.get("times_reinforced", 0) + 1
        else:
            self._states[memory_id] = {
                "memory_id": memory_id,
                "last_accessed": datetime.now(timezone.utc).isoformat(),
                "access_count": 0, "times_reinforced": 1,
                "importance": 5.0, "emotional_arousal": 0.0,
                "emotional_valence": 0.0, "belief_confidence": 0.5,
                "related_count": 0,
            }
        self._dirty = True

    def flush(self) -> None:
        if self._dirty:
            os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
            import tempfile
            dir_name = os.path.dirname(self._cache_path) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._states, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self._cache_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            self._dirty = False

    def calculate_strength(self, memory_id: str, current_time: Optional[datetime] = None) -> float:
        s = self._states.get(memory_id)
        if not s or not s.get("last_accessed"):
            return 0.0
        ds = DecayState(
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
        return ds.calculate_strength(now, emotion_slowdown=self._emotion_slowdown)

    def get_threshold(self, memory_id: str) -> float:
        s = self._states.get(memory_id, {})
        importance = s.get("importance", 5.0)
        imp_bonus = (importance / 10.0) * 0.3
        return round(self._base_threshold + imp_bonus, 4)

    @staticmethod
    def _is_immortal(mem: Dict) -> bool:
        importance = mem.get("importance", 0)
        if importance >= 9:
            return True
        meta = mem.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if meta.get("source") == "consolidated" and meta.get("memory_type") == "semantic":
            return True
        return False

    def evaluate_all(self, memories: List[Dict], overwriter: Optional["MemoryOverwriter"] = None) -> Dict[str, Any]:
        to_forget, to_keep = [], []
        strengths: Dict[str, float] = {}
        now = datetime.now(timezone.utc)

        for mem in memories:
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

        total = len(memories)
        return {
            "to_forget": to_forget,
            "to_keep": to_keep,
            "report": {
                "total_evaluated": total,
                "to_forget_count": len(to_forget),
                "to_keep_count": len(to_keep),
                "forget_ratio": round(len(to_forget) / total, 3) if total else 0,
                "avg_strength": round(sum(strengths.values()) / len(strengths), 3) if strengths else 0,
                "strengths": strengths,
            },
        }

    def execute_forgetting(self, to_forget_ids: List[str], memory_forget_func: Any) -> int:
        deleted = 0
        for mid in to_forget_ids:
            try:
                memory_forget_func(mid)
                self._states.pop(mid, None)
                deleted += 1
            except Exception:
                pass
        self._dirty = True
        self.flush()
        return deleted

    def _sync_from_memory(self, mem: Dict) -> None:
        mid = mem.get("id", "")
        if not mid:
            return
        if mid in self._states:
            # Sync importance for existing entries (may have been updated by consolidation)
            self._states[mid]["importance"] = float(mem.get("importance", 5))
            return
        meta = mem.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        self._states[mid] = {
            "memory_id": mid,
            "last_accessed": mem.get("created_at", datetime.now(timezone.utc).isoformat()),
            "access_count": 0, "times_reinforced": 0,
            "importance": float(mem.get("importance", 5)),
            "emotional_arousal": meta.get("emotional_arousal", 0.0),
            "emotional_valence": meta.get("emotional_valence", 0.0),
            "belief_confidence": meta.get("belief_confidence", 0.5),
            "related_count": len(meta.get("related_memory_ids", [])),
            "memory_type": meta.get("memory_type", "episodic"),
        }


class MemoryOverwriter:
    """Handle memory conflicts by superseding old beliefs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self._conflict_threshold = self.cfg.get("collision_threshold", 0.7)
        self._downgrade_factor = self.cfg.get("overwriting_downgrade_factor", 0.3)

    def handle_conflict(self, old_belief: Dict, new_evidence: Dict) -> Dict:
        if not old_belief or not new_evidence:
            return {"action": "no_conflict"}

        old_text = old_belief.get("text", "").lower()
        new_text = new_evidence.get("text", "").lower()
        if not old_text or not new_text:
            return {"action": "no_conflict"}

        if not self._detect_conflict(old_text, new_text):
            return {"action": "no_conflict"}

        old_meta = old_belief.get("metadata", {})
        if isinstance(old_meta, str):
            try:
                old_meta = json.loads(old_meta)
            except Exception:
                old_meta = {}
        old_conf = old_meta.get("belief_confidence", 0.5) if isinstance(old_meta, dict) else 0.5
        new_conf = new_evidence.get("metadata", {}).get("belief_confidence", 0.5)

        if new_conf > old_conf:
            return self._supersede(old_belief, new_evidence)
        else:
            return {"action": "keep_old", "reason": "old has higher confidence"}

    def check_and_handle(self, candidate: Dict, all_memories: List[Dict]) -> Dict:
        meta = candidate.get("metadata", {})
        if isinstance(meta, dict) and meta.get("superseded"):
            return {"superseded": True}
        for other in all_memories:
            if other.get("id", "") == candidate.get("id", ""):
                continue
            if self._detect_conflict(candidate.get("text", "").lower(), other.get("text", "").lower()):
                return self.handle_conflict(other, candidate)
        return {"superseded": False}

    @staticmethod
    def _detect_conflict(text_a: str, text_b: str) -> bool:
        from hms.utils.text import tokenize
        tokens_a = set(tokenize(text_a))
        tokens_b = set(tokenize(text_b))
        if not tokens_a or not tokens_b:
            return False
        overlap = tokens_a & tokens_b
        overlap_ratio = len(overlap) / max(len(tokens_a), len(tokens_b), 1)
        if overlap_ratio < 0.3:
            return False
        negation_words = {"不", "没", "非", "否", "not", "no", "never", "don't", "doesn't"}
        has_neg_a = any(w in text_a for w in negation_words)
        has_neg_b = any(w in text_b for w in negation_words)
        if has_neg_a != has_neg_b and overlap_ratio > 0.4:
            return True
        return False

    def _supersede(self, old: Dict, new: Dict) -> Dict:
        old_meta = old.get("metadata", {})
        if isinstance(old_meta, str):
            try:
                old_meta = json.loads(old_meta)
            except Exception:
                old_meta = {}
        if not isinstance(old_meta, dict):
            old_meta = {}
        old_meta["superseded"] = True
        old_meta["superseded_at"] = datetime.now(timezone.utc).isoformat()
        old_meta["superseded_by"] = new.get("id", "unknown")
        old_meta["belief_confidence"] = round(
            old_meta.get("belief_confidence", 0.5) * self._downgrade_factor, 3
        )
        return {
            "action": "superseded",
            "old_id": old.get("id", ""),
            "new_id": new.get("id", ""),
            "updated_metadata": old_meta,
        }
