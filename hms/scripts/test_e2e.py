"""
HMS v2 — End-to-End Test Suite.

Validates the full pipeline: perception -> collision -> context -> consolidation -> forgetting.
Tests both LLM and heuristic paths. LLM calls are mocked when no API key is available.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ======================================================================
# Mock LLM responses
# ======================================================================

MOCK_PERCEIVE_RESPONSE = json.dumps({
    "entities": [{"name": "Python", "type": "tool"}],
    "emotion": {"valence": 0.5, "arousal": 0.3, "primary_emotion": "happy", "evidence": "用户说喜欢"},
    "intent": {"primary": "陈述", "confidence": 0.9},
    "importance": 7,
    "importance_reason": "用户表达了偏好",
    "category": "preference",
    "topics": ["编程"],
    "key_facts": ["用户喜欢Python"],
    "should_remember": True
})

MOCK_COLLIDE_RESPONSE = json.dumps({
    "contradictions": [],
    "reinforcements": [{"existing_id": "m1", "confidence_delta": 0.05, "reason": "same tool"}],
    "associations": [{"existing_id": "m1", "relation_type": "co_mentioned", "confidence": 0.8, "reason": "Python"}],
    "inferences": []
})

MOCK_CONSOLIDATE_RESPONSE = json.dumps({
    "summary": "用户讨论了Python编程",
    "key_decisions": ["使用Python"],
    "preferences_revealed": ["喜欢Python"],
    "emotional_moments": [],
    "entities_mentioned": ["Python"],
    "topics": ["编程"],
    "thinking_patterns": ["注重效率"],
    "timeline_entry": {"date": "2024-01-01", "topic": "编程", "summary": "讨论Python", "importance": 7}
})

MOCK_FINGERPRINT_RESPONSE = json.dumps({
    "thinking_patterns": ["注重效率", "喜欢简洁"],
    "core_preferences": ["Python"],
    "emotional_triggers": {"positive": ["成功"], "negative": ["bug"]},
    "focus_areas": ["编程"],
    "communication_style": "直接",
    "values": ["效率"],
    "recent_goals": ["学习新技术"],
    "personality_notes": "务实",
    "confidence": 0.8,
    "version": 1
})


def mock_llm_call(prompt, max_tokens=1000, temperature=0.1):
    """Return mock responses based on prompt content."""
    if "感知" in prompt or "perceive" in prompt.lower() or "对话" in prompt:
        return MOCK_PERCEIVE_RESPONSE
    if "碰撞" in prompt or "collide" in prompt.lower() or "对比" in prompt:
        return MOCK_COLLIDE_RESPONSE
    if "压缩" in prompt or "consolidate" in prompt.lower() or "摘要" in prompt:
        return MOCK_CONSOLIDATE_RESPONSE
    if "指纹" in prompt or "fingerprint" in prompt.lower() or "画像" in prompt:
        return MOCK_FINGERPRINT_RESPONSE
    return MOCK_PERCEIVE_RESPONSE


def run_tests():
    """Run all HMS v3 self-tests."""
    passed = 0
    failed = 0
    errors = []

    def test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            print(f"  OK {name}")
        except Exception as e:
            failed += 1
            err_str = repr(e) if str(e) else type(e).__name__
            errors.append((name, err_str))
            print(f"  FAIL {name}: {err_str}")

    print("=" * 60)
    print("HMS v3 -- End-to-End Test Suite")
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
    test("Token estimation", _test_token_estimate)
    test("Circuit breaker", _test_circuit_breaker)
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

    # --- Embedding Cache ---
    print("[Embedding Cache]")
    test("Embed single", _test_embed_single)
    test("Similarity", _test_embed_similarity)
    test("Find similar", _test_embed_find)
    test("Prefilter", _test_embed_prefilter)
    print()

    # --- Context Manager ---
    print("[Context Manager]")
    test("Pending queue", _test_pending_queue)
    test("Pop all pending (atomic)", _test_pop_all_pending)
    test("Fingerprint CRUD", _test_fp_crud)
    test("Fingerprint cap", _test_fp_cap)
    test("Timeline CRUD", _test_timeline_crud)
    test("Context composition", _test_context_compose)
    test("Token estimation", _test_token_est)
    test("Budget minimum floor", _test_budget_floor)
    print()

    # --- Consolidation ---
    print("[Consolidation Engine]")
    test("Replay selection", _test_replay_select)
    test("Conversation compression", _test_compress)
    test("Relation discovery", _test_relations)
    test("Fallback general topic", _test_fallback_general)
    print()

    # --- Forgetting ---
    print("[Forgetting Engine]")
    test("Access tracking", _test_forget_access)
    test("Dirty flag batching", _test_forget_dirty_flag)
    test("Strength calculation", _test_forget_strength)
    test("Immortal guard", _test_immortal)
    test("Consistency sync", _test_forget_sync)
    print()

    # --- Memory Manager ---
    print("[Memory Manager]")
    test("Init", _test_mm_init)
    test("On message received", _test_mm_received)
    test("On message sent", _test_mm_sent)
    test("Process pending", _test_mm_process)
    test("Consolidate", _test_mm_consolidate)
    test("Forget", _test_mm_forget)
    test("Tier support", _test_mm_tier)
    print()

    # --- LLM Mocked Integration ---
    print("[LLM Mocked Integration]")
    test("Mocked perceive", _test_mocked_perceive)
    test("Mocked collision", _test_mocked_collision)
    test("Mocked consolidation", _test_mocked_consolidation)
    print()

    # --- Summary ---
    print("=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} FAILED")
        for name, err in errors:
            print(f"  FAIL {name}: {err}")
    else:
        print(" -- ALL PASSED")
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
    assert "circuit_open" in s


def _test_token_estimate():
    from hms.scripts.utils import estimate_tokens
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("你好世界测试") > estimate_tokens("hello test")
    assert estimate_tokens("") == 0


def _test_circuit_breaker():
    from hms.scripts.llm_analyzer import LLMAnalyzer
    a = LLMAnalyzer()
    assert a._consecutive_failures == 0
    a._trip_circuit()
    assert a._circuit_open_until > 0
    s = a.get_stats()
    assert s["circuit_open"] is True


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


def _test_embed_single():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.embed_cache import EmbeddingCache
        cache = EmbeddingCache({"cache_dir": d})
        vec = cache.embed("test text")
        assert len(vec) == 256
        stats = cache.get_stats()
        assert stats["encoder_type"] in ("char_ngram", "sentence-transformers")


def _test_embed_similarity():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.embed_cache import EmbeddingCache
        cache = EmbeddingCache({"cache_dir": d})
        sim = cache.similarity("Python code", "Python programming")
        assert 0.0 <= sim <= 1.0


def _test_embed_find():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.embed_cache import EmbeddingCache
        cache = EmbeddingCache({"cache_dir": d})
        candidates = [{"text": "Python语言", "id": "a"}, {"text": "天气好", "id": "b"}]
        results = cache.find_similar("Python", candidates, top_k=2, threshold=0.05)
        assert len(results) >= 1


def _test_embed_prefilter():
    from hms.scripts.embed_cache import EmbeddingCache, prefilter_for_collision
    with tempfile.TemporaryDirectory() as d:
        cache = EmbeddingCache({"cache_dir": d})
        memories = [{"text": "Python编程", "id": "m1"}, {"text": "天气好", "id": "m2"}]
        filtered = prefilter_for_collision("Python代码", memories, cache, similarity_threshold=0.1)
        assert isinstance(filtered, list)


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


def _test_pop_all_pending():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.context_manager import ContextManager
        cm = ContextManager({"pending_path": os.path.join(d, "p.jsonl")})
        cm.enqueue("a", "b")
        cm.enqueue("c", "d")
        entries = cm.pop_all_pending()
        assert len(entries) == 2
        assert cm.get_pending_count() == 0
        entries2 = cm.pop_all_pending()
        assert len(entries2) == 0


def _test_fp_cap():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.context_manager import ContextManager
        cm = ContextManager({"pending_path": os.path.join(d, "p.jsonl")})
        for i in range(15):
            cm.update_fingerprint({"thinking_patterns": [f"pattern_{i}"]})
        fp = cm.get_fingerprint()
        assert len(fp.get("thinking_patterns", [])) <= 10


def _test_fp_crud():
    with tempfile.TemporaryDirectory() as d:
        from hms.scripts.context_manager import ContextManager
        cm = ContextManager({"pending_path": os.path.join(d, "p.jsonl")})
        cm.update_fingerprint({"thinking_patterns": ["analytical"], "core_preferences": ["Python"]})
        fp = cm.get_fingerprint()
        assert "analytical" in fp.get("thinking_patterns", [])
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
    from hms.scripts.utils import estimate_tokens
    from hms.scripts.context_manager import ContextManager
    assert estimate_tokens("hello world") > 0
    assert estimate_tokens("你好世界") > 0
    assert ContextManager.truncate_to_tokens("hello world", 1) != "hello world"


def _test_budget_floor():
    from hms.scripts.context_manager import ContextManager
    with tempfile.TemporaryDirectory() as d:
        cm = ContextManager({"pending_path": os.path.join(d, "p.jsonl")})
        # Test with a very small window to trigger minimum floor
        budget = cm.estimate_token_budget(model_context_window=5000)
        assert budget["cognitive_fingerprint"] >= 300
        assert budget["injected_memories"] >= 500
        assert budget["recent_turns"] >= 1000


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


def _test_fallback_general():
    from hms.scripts.consolidation import ConsolidationEngine
    e = ConsolidationEngine()
    convs = [{"user": "今天心情不错", "assistant": "很高兴听到这个"}]
    r = e.compress_conversations(convs)
    assert r is not None
    assert "general" in r.get("topics", [])


def _test_forget_access():
    import tempfile
    path = tempfile.mktemp(suffix=".json")
    try:
        from hms.scripts.forgetting import ForgettingEngine
        fe = ForgettingEngine({"decay_cache_path": path})
        fe.update_on_access("m1")
        assert "m1" in fe._states
        assert fe._dirty is True
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _test_forget_dirty_flag():
    import tempfile
    path = tempfile.mktemp(suffix=".json")
    try:
        from hms.scripts.forgetting import ForgettingEngine
        fe = ForgettingEngine({"decay_cache_path": path})
        fe.update_on_access("m1")
        assert fe._dirty is True
        fe.flush()
        assert fe._dirty is False
        fe.update_on_access("m2")
        assert fe._dirty is True
        fe.flush()
        assert fe._dirty is False
        assert "m2" in fe._states
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


def _test_forget_sync():
    import tempfile
    path = tempfile.mktemp(suffix=".json")
    try:
        from hms.scripts.forgetting import ForgettingEngine
        fe = ForgettingEngine({"decay_cache_path": path})
        fe.update_on_access("m1")
        fe.update_on_access("m2")
        # Sync with memories where m2 was deleted
        memories = [{"id": "m1", "importance": 7, "metadata": {}}, {"id": "m3", "importance": 5, "metadata": {}}]
        report = fe.sync_consistency(memories)
        assert report["orphaned_removed"] == 1  # m2 removed
        assert report["missing_added"] == 1  # m3 added
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _test_mm_init():
    from hms.scripts.memory_manager import MemoryManager
    mgr = MemoryManager({"cache_dir": tempfile.mkdtemp()})
    assert mgr.perception is not None
    assert mgr.embed_cache is not None


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


def _test_mm_tier():
    from hms.scripts.memory_manager import MemoryManager
    import json
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(config_path) as f:
        base_cfg = json.load(f)
    for tier in ["32k", "128k", "256k", "1M"]:
        cfg = MemoryManager._apply_tier(dict(base_cfg), tier)
        assert cfg.get("model_context_window", 0) > 0


def _test_mocked_perceive():
    """Test perception with mocked LLM calls."""
    from hms.scripts.llm_analyzer import LLMAnalyzer
    a = LLMAnalyzer()
    with patch.object(a, '_call_llm', side_effect=mock_llm_call):
        result = a.perceive("我喜欢Python")
        assert result is not None
        assert result.get("should_remember") is True


def _test_mocked_collision():
    """Test collision with mocked LLM calls."""
    from hms.scripts.llm_analyzer import LLMAnalyzer
    a = LLMAnalyzer()
    call_log = []

    def mock_fn(prompt, max_tokens=1000, temperature=0.1):
        call_log.append("called")
        return MOCK_COLLIDE_RESPONSE

    with patch.object(a, '_call_llm', side_effect=mock_fn):
        result = a.collide(
            {'text_for_store': 'Python不错'},
            [{'id': 'm1', 'text': '用户用Python'}]
        )
    assert result is not None, f"result is None, call_log={call_log}"
    assert "contradictions" in result, f"no contradictions in result: {result}"


def _test_mocked_consolidation():
    """Test consolidation with mocked LLM calls."""
    from hms.scripts.llm_analyzer import LLMAnalyzer
    a = LLMAnalyzer()
    call_log = []

    def mock_fn(prompt, max_tokens=1000, temperature=0.1):
        call_log.append("called")
        return MOCK_CONSOLIDATE_RESPONSE

    with patch.object(a, '_call_llm', side_effect=mock_fn):
        result = a.consolidate(
            [{'user': 'Python好', 'assistant': '是的'}],
            {'thinking_patterns': []}
        )
    assert result is not None, f"result is None, call_log={call_log}"
    assert "summary" in result, f"no summary in result: {result}"


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
