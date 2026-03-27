"""
HMS v2 — Shared data models.

Extends v1 models with cognitive fingerprint and topic timeline support.
Uses only Python standard library.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------  
# Enums  
# ---------------------------------------------------------------------------  

class MemoryType(Enum):
    """Four memory types mirroring cognitive science taxonomy."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    PROSPECTIVE = "prospective"


class MemorySource(Enum):
    """Where the memory was acquired."""
    DIRECT_EXPERIENCE = "direct_experience"
    INFERRED = "inferred"
    REPORTED = "reported"
    CONSOLIDATED = "consolidated"


class BeliefStrength(Enum):
    """Epistemic status of a belief."""
    CERTAIN = "certain"
    LIKELY = "likely"
    UNCERTAIN = "uncertain"
    CONTRADICTED = "contradicted"


# ---------------------------------------------------------------------------  
# Dataclasses  
# ---------------------------------------------------------------------------  

@dataclass
class MemoryBelief:
    """A belief attached to a memory, tracking evidence and confidence."""
    content: str
    strength: BeliefStrength = BeliefStrength.UNCERTAIN
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    last_evaluated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update_confidence(self) -> float:
        ev = len(self.evidence)
        con = len(self.contradictions)
        total = ev + con
        if total == 0:
            self.confidence = 0.5
        else:
            self.confidence = ev / total
        strength_mult = {
            BeliefStrength.CERTAIN: 1.0,
            BeliefStrength.LIKELY: 0.8,
            BeliefStrength.UNCERTAIN: 0.5,
            BeliefStrength.CONTRADICTED: 0.1,
        }
        self.confidence = round(min(1.0, self.confidence * strength_mult[self.strength]), 4)
        self.last_evaluated = datetime.now(timezone.utc)
        if self.confidence >= 0.9:
            self.strength = BeliefStrength.CERTAIN
        elif self.confidence >= 0.6:
            self.strength = BeliefStrength.LIKELY
        elif self.confidence >= 0.3:
            self.strength = BeliefStrength.UNCERTAIN
        else:
            self.strength = BeliefStrength.CONTRADICTED
        return self.confidence


@dataclass
class EmotionalTrace:
    """Minimal affect tag carried by a memory."""
    valence: float = 0.0
    arousal: float = 0.0
    primary_emotion: str = "neutral"
    context: str = ""

    def __post_init__(self):
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))


@dataclass
class MemoryMeta:
    """Cognitive metadata envelope for memory-lancedb-pro entries."""
    belief_strength: str = "uncertain"
    belief_confidence: float = 0.5
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.0
    primary_emotion: str = "neutral"
    memory_type: str = "episodic"
    source: str = "direct_experience"
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    related_memory_ids: List[str] = field(default_factory=list)
    derived_from: List[str] = field(default_factory=list)
    access_count: int = 0
    times_reinforced: int = 0
    source_reliability: float = 1.0
    abstract_version: str = ""

    def to_metadata_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def from_metadata_json(json_str: str) -> "MemoryMeta":
        data = json.loads(json_str)
        return MemoryMeta(**data)

    @staticmethod
    def from_belief_and_trace(
        belief: MemoryBelief,
        trace: EmotionalTrace,
        memory_type: MemoryType = MemoryType.EPISODIC,
        source: MemorySource = MemorySource.DIRECT_EXPERIENCE,
        entities: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        source_reliability: float = 1.0,
    ) -> "MemoryMeta":
        return MemoryMeta(
            belief_strength=belief.strength.value,
            belief_confidence=belief.confidence,
            emotional_valence=trace.valence,
            emotional_arousal=trace.arousal,
            primary_emotion=trace.primary_emotion,
            memory_type=memory_type.value,
            source=source.value,
            entities=entities or [],
            topics=topics or [],
            source_reliability=source_reliability,
        )


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
        emo_mult = 1.0 + self.emotional_arousal * emotion_slowdown
        rein_mult = 1.0 + self.times_reinforced * reinforcement_weight
        conf_mult = 1.0 + self.belief_confidence * 0.5
        rel_mult = 1.0 + self.related_count * 0.05
        strength = self.importance * emo_mult * rein_mult * conf_mult * rel_mult * time_factor
        return round(max(0.0, strength), 4)


# ---------------------------------------------------------------------------  
# v2: Cognitive Structures  
# ---------------------------------------------------------------------------  

@dataclass
class CognitiveFingerprint:
    """
    Dynamic cognitive profile of the user.
    Compressed representation of long-term interaction patterns.
    Target size: < 2000 tokens.
    """
    thinking_patterns: List[str] = field(default_factory=list)
    core_preferences: List[str] = field(default_factory=list)
    emotional_triggers: Dict[str, List[str]] = field(
        default_factory=lambda: {"positive": [], "negative": []}
    )
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


@dataclass
class TopicTimeline:
    """
    Per-topic timeline of important events.
    Keeps recent entries per topic, oldest get merged into summaries.
    """
    topic: str = ""
    entries: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""  # merged historical summary
    last_updated: str = ""
    total_entries_merged: int = 0

    def add_entry(self, date: str, summary: str, importance: int = 5) -> None:
        self.entries.append({
            "date": date,
            "summary": summary,
            "importance": importance,
        })
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def prune(self, max_entries: int = 10) -> int:
        """Keep only top entries by importance, merge rest into summary."""
        if len(self.entries) <= max_entries:
            return 0
        sorted_entries = sorted(self.entries, key=lambda e: e.get("importance", 0), reverse=True)
        keep = sorted_entries[:max_entries]
        merge = sorted_entries[max_entries:]
        # Build summary from merged entries
        merged_summaries = [e.get("summary", "") for e in merge if e.get("summary")]
        if merged_summaries:
            self.summary = (self.summary + " " + " | ".join(merged_summaries)).strip()
            self.total_entries_merged += len(merge)
        self.entries = keep
        return len(merge)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "entries": self.entries,
            "summary": self.summary,
            "last_updated": self.last_updated,
            "total_entries_merged": self.total_entries_merged,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TopicTimeline":
        t = TopicTimeline()
        t.topic = data.get("topic", "")
        t.entries = data.get("entries", [])
        t.summary = data.get("summary", "")
        t.last_updated = data.get("last_updated", "")
        t.total_entries_merged = data.get("total_entries_merged", 0)
        return t


@dataclass
class CompressedSummary:
    """A compressed block of conversation history."""
    original_turn_ids: List[str] = field(default_factory=list)
    summary_text: str = ""
    key_decisions: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    emotional_moments: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    token_estimate: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CompressedSummary":
        return CompressedSummary(**data)


# ---------------------------------------------------------------------------  
# Self-test  
# ---------------------------------------------------------------------------  

def _self_test():
    """Run: python -m hms.scripts.models"""
    now = datetime.now(timezone.utc)

    # Test MemoryBelief
    b = MemoryBelief(content="用户喜欢咖啡", confidence=0.5)
    b.evidence.append("用户点了拿铁")
    b.update_confidence()
    assert 0.0 <= b.confidence <= 1.0
    print(f"[Belief] confidence={b.confidence} strength={b.strength.value}")

    # Test CognitiveFingerprint
    fp = CognitiveFingerprint.default()
    fp.thinking_patterns.append("注重细节")
    fp.core_preferences.append("喜欢简洁的代码")
    d = fp.to_dict()
    fp2 = CognitiveFingerprint.from_dict(d)
    assert fp2.thinking_patterns == ["注重细节"]
    print(f"[Fingerprint] OK, patterns={len(fp2.thinking_patterns)}")

    # Test TopicTimeline
    tl = TopicTimeline(topic="项目A")
    tl.add_entry("2024-01-01", "项目启动", 8)
    tl.add_entry("2024-01-15", "第一次里程碑", 7)
    tl.prune(max_entries=1)
    assert len(tl.entries) == 1
    assert tl.total_entries_merged == 1
    print(f"[Timeline] entries={len(tl.entries)} merged={tl.total_entries_merged}")

    print("✓ All self-tests passed.")


if __name__ == "__main__":
    _self_test()
