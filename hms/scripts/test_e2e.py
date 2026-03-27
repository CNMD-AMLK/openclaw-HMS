"""
HMS v2 — End-to-End Test Suite.

Validates the full pipeline: perception → collision → context → consolidation → forgetting.
Tests both LLM and heuristic paths.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def run_tests():
    """Run all HMS v2 self-tests."""
    passed = 0
    failed = 0
    errors = []

    def test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  ✗ {name}: {e}")

    print("=" * 60)
    print("HMS v2 — End-to-End Test Suite")
    print("=" * 60)
    print()

    # --- Models ---
    print("[Models]")
    test("MemoryBelief", _test_belief)
    test("CognitiveFingerprint", _test_fingerprint)
    test("TopicTimeline", _test_timeline)
    test("DecayState", _test_decay)
    print()

    # --- LLM Analyzer ---
    print("[LLM Analyzer]")
    test("Fallback perceive", _test_fallback_perceive)
    test("JSON parsing", _test_json_parse)
    test("Stats tracking", _test_stats)
    print()

    # --- Perception ---
    print("[Perception Engine]")
    test("Lite mode", _test_perception_lite)
    test("Full mode (fallback)", _test_perception_full)
    print()

    # --- Collision ---
    print("[Collision Engine]")
    test("Heuristic collision", _test_collision)
    test("Execute results", _test_collision_exec)
    print()

    # --- Context Manager ---
    print("[Context Manager]")
    test("Pending queue", _test_pending_queue)
    test("Fingerprint CRUD", _test_fp_crud)
    test("Timeline CRUD", _test_timeline_crud)
    test("Context composition", _test_context_compose)
    test("Token estimation", _test_token_est)
    print()

    # --- Consolidation ---
    print("[Consolidation Engine]")
    test("Replay selection", _test_replay_select)
    test("Conversation compression", _test_compress)
    test("Relation discovery", _test_relations)
    print()

    # --- Forgetting ---
    print("[Forgetting Engine]")
    test("Access tracking", _test_forget_access)
    test("Strength calculation", _test_forget_strength)
    test("Immortal guard", _test_immortal)
    print()

    # --- Memory Manager ---
    print("[Memory Manager]")
    test("Init", _test_mm_init)
    test("On message received", _test_mm_received)
    test("On message sent", _test_mm_sent)
    test("Process pending", _test_mm_process)
    test("Consolidate", _test_mm_consolidate)
    test("Forget", _test_mm_forget)
    print()

    # --- Summary ---
    print("=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} FAILED")
        for name, err in errors:
            print(f"  ✗ {name}: {err}")
    else:
        print(" — ALL PASSED ✓")
    print("=" * 60)

    return failed == 0


# ======================================================================
# Test implementations
# ======================================================================

def _test_belief():
    from hms.scripts.models import MemoryBelief, BeliefStrength
    b = MemoryBelief(content="test", confidence=0.5)
    b.evidence.append("ev1")
    b.update_confidence()
    assert 0.0 <= b.confidence <= 1.0


def _test_fingerprint():
    from hms.scripts.models import CognitiveFingerprint
    fp = CognitiveFingerprint.default()
    fp.thinking_patterns.append("detail-oriented")
    d = fp.to_dict()
    fp2 = CognitiveFingerprint.from_dict(d)
    assert fp2.thinking_patterns == ["detail-oriented"]


def _test_timeline():
    from hms.scripts.models import TopicTimeline
    tl = TopicTimeline(topic="test")
    tl.add_entry("2024-01-01", "event1", 8)
    tl.add_entry("2024-01-02", "event2", 3)
    tl.prune(max_entries=1)
    assert len(tl.entries) == 1
    assert tl.total_entries_merged == 1


def _test_decay():
    from hms.scripts.models import DecayState
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    d = DecayState(
        memory_id="m1",
        last_accessed=(now - timedelta(hours=24)).isoformat(),
        importance=7.0,
    )
    s = d.calculate_strength(now)
    assert s > 0


def _test_fallback_perceive():
    from hms.scripts.llm_analyzer import LLMAnalyzer
    a = LLMAnalyzer()
    r = a.fallback_perceive("我决定用Python")
    assert r["category"] == "decision"
    assert r["importance"] >= 5


def _test_json_parse():
    from hms.scripts.llm_analyzer import LLMAnalyzer
    a = LLMAnalyzer()
    assert a._parse_json_response('```json\n{"k":"v"}\n```') == {"k": "v"}
    assert a._parse_json_response('{"k":"v"}') == {"k": "v"}


def _test_stats():
    from hms.scripts.llm_analyzer import LLMAnalyzer
    a = LLMAnalyzer()
    s = a.get_stats()
    assert "call_count" in s


def _test_perception_lite():
    from hms.scripts.perception import PerceptionEngine
    e = PerceptionEngine({"llm_perception_mode": "lite"})
    r = e.analyze("帮我安装Docker")
    assert r["intent"]["primary"] == "请求"


def _test_perception_full():
    from hms.scripts.perception import PerceptionEngine
    e = PerceptionEngine({"llm_perception_mode": "full"})
    r = e.analyze("今天天气不错")
    assert "analysis_method" in r


def _test_collision():
    from hms.scripts.collision import CollisionEngine
    e = CollisionEngine()
    p = {"text_for_store": "Python很好", "entities": [{"name": "Python", "type": "tool"}]}
    mems = [{"id": "m1", "text": "Python不错", "metadata": {"entities": ["Python"]}}]
    r = e.collide(p, mems)
    assert "contradictions" in r


def _test_collision_exec():
    from hms.scripts.collision import CollisionEngine
    e = CollisionEngine()
    stored = []
    r = e.execute_results(
        {"new_insights": [], "associations": [{"existing_id": "m1", "confidence": 0.7, "relation_type": "test", "reason": "test"}]},
        gm_record_func=lambda **kw: stored.append(kw),
    )
    assert r["graph_edges"] >= 0


def _test_pending_queue():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.context_manager import ContextManager
        cm = ContextManager({"pending_path": os.path.join(d, "p.jsonl")})
        cm.enqueue("hello", "hi")
        cm.enqueue("bye", "goodbye")
        assert cm.get_pending_count() == 2
        entries = cm.read_pending()
        assert len(entries) == 2
        cm.clear_pending()
        assert cm.get_pending_count() == 0


def _test_fp_crud():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.context_manager import ContextManager
        cm = ContextManager({"pending_path": os.path.join(d, "p.jsonl")})
        cm.update_fingerprint({"thinking_patterns": ["analytical"], "core_preferences": ["Python"]})
        fp = cm.get_fingerprint()
        assert "analytical" in fp.get("thinking_patterns", [])
        # Merge
        cm.update_fingerprint({"thinking_patterns": ["creative"]})
        fp2 = cm.get_fingerprint()
        assert "analytical" in fp2["thinking_patterns"]
        assert "creative" in fp2["thinking_patterns"]


def _test_timeline_crud():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.context_manager import ContextManager
        cm = ContextManager({"pending_path": os.path.join(d, "p.jsonl")})
        cm.update_timelines([{"topic": "AI", "date": "2024-01-01", "summary": "started", "importance": 8}])
        tl = cm.get_timelines()
        assert "AI" in tl


def _test_context_compose():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.context_manager import ContextManager
        cm = ContextManager({"pending_path": os.path.join(d, "p.jsonl")})
        cm.update_fingerprint({"thinking_patterns": ["fast"]})
        ctx = cm.compose_context(
            system_prompt="You are helpful",
            injected_memories=[{"text": "user likes Python", "importance": 7}],
            recent_turns=[{"user": "hi", "assistant": "hello"}],
        )
        assert "cognitive_fingerprint" in ctx
        assert "topic_timelines" in ctx
        assert "compressed_summaries" in ctx
        assert ctx["total_tokens_estimated"] > 0


def _test_token_est():
    from hms.scripts.context_manager import ContextManager
    assert ContextManager.estimate_tokens("hello world") > 0
    assert ContextManager.estimate_tokens("你好世界") > 0
    assert ContextManager.truncate_to_tokens("hello world", 1) != "hello world"


def _test_replay_select():
    from hms.scripts.consolidation import ConsolidationEngine
    e = ConsolidationEngine()
    mems = [
        {"id": f"m{i}", "importance": i, "metadata": {"emotional_arousal": 0.5}}
        for i in range(10)
    ]
    selected = e.select_for_replay(mems, max_count=3)
    assert len(selected) == 3


def _test_compress():
    from hms.scripts.consolidation import ConsolidationEngine
    e = ConsolidationEngine()
    convs = [{"user": "hi", "assistant": "hello"}, {"user": "how are you", "assistant": "fine"}]
    r = e.compress_conversations(convs)
    assert r is not None
    assert r["original_turn_count"] == 2


def _test_relations():
    from hms.scripts.consolidation import ConsolidationEngine
    e = ConsolidationEngine()
    mems = [
        {"id": "a", "metadata": {"entities": ["Python"], "topics": ["coding"]}},
        {"id": "b", "metadata": {"entities": ["Python"], "topics": ["AI"]}},
    ]
    rels = e.discover_relations(mems)
    assert len(rels) >= 1


def _test_forget_access():
    import tempfile
    path = tempfile.mktemp(suffix=".json")
    try:
        from hms.scripts.forgetting import ForgettingEngine
        fe = ForgettingEngine({"decay_cache_path": path})
        fe.update_on_access("m1")
        assert "m1" in fe._states
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _test_forget_strength():
    import tempfile
    path = tempfile.mktemp(suffix=".json")
    try:
        from hms.scripts.forgetting import ForgettingEngine
        fe = ForgettingEngine({"decay_cache_path": path})
        fe.update_on_access("m1")
        s = fe.calculate_strength("m1")
        assert s > 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _test_immortal():
    from hms.scripts.forgetting import ForgettingEngine
    assert ForgettingEngine._is_immortal({"importance": 10, "metadata": {}})
    assert not ForgettingEngine._is_immortal({"importance": 3, "metadata": {}})


def _test_mm_init():
    from hms.scripts.memory_manager import MemoryManager
    mgr = MemoryManager({"cache_dir": tempfile.mkdtemp()})
    assert mgr.perception is not None


def _test_mm_received():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.memory_manager import MemoryManager
        mgr = MemoryManager({"cache_dir": d})
        result = mgr.on_message_received("你好")
        assert "perception" in result
        assert "context" in result


def _test_mm_sent():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.memory_manager import MemoryManager
        mgr = MemoryManager({"cache_dir": d})
        mgr.on_message_sent("你好", "你好！")
        assert mgr.context.get_pending_count() == 1


def _test_mm_process():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.memory_manager import MemoryManager
        mgr = MemoryManager({"cache_dir": d})
        mgr.on_message_sent("我决定用Python", "好的")
        result = mgr.process_pending()
        assert result["processed"] == 1


def _test_mm_consolidate():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.memory_manager import MemoryManager
        mgr = MemoryManager({"cache_dir": d})
        mgr.on_message_sent("hello", "hi")
        result = mgr.consolidate()
        assert "compressed" in result


def _test_mm_forget():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.memory_manager import MemoryManager
        mgr = MemoryManager({"cache_dir": d})
        result = mgr.forget()
        assert "evaluated" in result


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
