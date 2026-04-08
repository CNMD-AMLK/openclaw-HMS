"""
HMS v5 — Data models (adapted from hms-v4 scripts/models.py).
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class DecayState:
    """Per-memory decay bookkeeping."""
    memory_id: str = ""
    last_accessed: str = ""
    access_count: int = 0
    times_reinforced: int = 0
    importance: float = 5.0
    emotional_arousal: float = 0.0
    emotional_valence: float = 0.0
    belief_confidence: float = 0.5
    related_count: int = 0

    def calculate_strength(
        self,
        current_time: datetime,
        *,
        emotion_slowdown: float = 2.0,
        reinforcement_weight: float = 0.15,
        half_life_hours: float = 168.0,
    ) -> float:
        if not self.last_accessed:
            return 0.0
        last = datetime.fromisoformat(self.last_accessed)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        hours_elapsed = max(0.0, (current_time - last).total_seconds() / 3600.0)
        time_factor = math.exp(-0.693147 * hours_elapsed / half_life_hours)

        emo_mult = 1.0 + min(self.emotional_arousal, 1.0) * min(emotion_slowdown, 2.0)
        rein_mult = 1.0 + min(self.times_reinforced, 3) * reinforcement_weight
        conf_mult = 1.0 + min(self.belief_confidence, 1.0) * 0.5
        rel_mult = 1.0 + min(self.related_count, 6) * 0.05

        strength = self.importance * emo_mult * rein_mult * conf_mult * rel_mult * time_factor
        return round(max(0.0, strength), 4)


@dataclass
class CognitiveFingerprint:
    """Dynamic cognitive profile of the user."""
    thinking_patterns: List[str] = field(default_factory=list)
    core_preferences: List[str] = field(default_factory=list)
    emotional_triggers: Dict[str, List[str]] = field(default_factory=lambda: {"positive": [], "negative": []})
    focus_areas: List[str] = field(default_factory=list)
    communication_style: str = ""
    values: List[str] = field(default_factory=list)
    recent_goals: List[str] = field(default_factory=list)
    personality_notes: str = ""
    last_updated: str = ""
    confidence: float = 0.5
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CognitiveFingerprint":
        return CognitiveFingerprint(**data)

    @staticmethod
    def default() -> "CognitiveFingerprint":
        return CognitiveFingerprint(
            last_updated=datetime.now(timezone.utc).isoformat(),
            version=1,
        )
