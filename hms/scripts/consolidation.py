"""
HMS v3.2 — Consolidation Engine.

LLM-driven memory consolidation: replay, compression, relation discovery.
Replaces v1's frequency-based clustering with semantic understanding.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .llm_analyzer import LLMAnalyzer
from .embed_cache import EmbeddingCache
from .utils import tokenize

logger = logging.getLogger(__name__)

# Emotional keyword patterns for fallback extraction
# Each pattern: (regex, intensity_label, is_positive)
_EMOTION_PATTERNS = [
    (re.compile(r'(太|很|非常|特别|超级|极其)\s*(高兴|开心|兴奋|激动|满意|舒服|棒|爽|快乐|幸福|感动|惊喜)'), "high", True),
    (re.compile(r'(太|很|非常|特别|超级|极其)\s*(愤怒|生气|烦躁|焦虑|紧张|害怕|恐惧|悲伤|难过|伤心|失望|痛苦|绝望|恶心|讨厌|烦|郁闷|无聊|孤独|寂寞|疲惫|累|困)'), "high", False),
    (re.compile(r'(!|！){2,}'), "high", None),  # neutral intensity marker
    (re.compile(r'(高兴|开心|兴奋|满意|舒服|棒|爽|快乐|幸福|感动|惊喜)'), "positive", True),
    (re.compile(r'(愤怒|生气|烦躁|焦虑|紧张|害怕|恐惧|悲伤|难过|伤心|失望|痛苦|绝望|恶心|讨厌|烦|郁闷|无聊|孤独|寂寞|疲惫|累|困)'), "negative", False),
]


class ConsolidationEngine:
    """
    Offline cognitive consolidation pipeline:
      1. Memory replay — re-evaluate memories against related ones
      2. Conversation compression — LLM summarization
      3. Topic timeline building
      4. Cognitive fingerprint updates
      5. Relation discovery
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self.llm = LLMAnalyzer(self.cfg)
        self.embed_cache: Optional[EmbeddingCache] = None

    # ==================================================================
    # 1. Memory Replay
    # ==================================================================

    def select_for_replay(
        self,
        memories: List[Dict[str, Any]],
        max_count: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Select memories for replay, prioritized by:
        priority = recency*0.3 + arousal*0.25 + importance*0.25
                 + low_confidence*0.1 + contradicted*0.1
        """
        now = datetime.now(timezone.utc)
        scored: List[tuple[float, Dict[str, Any]]] = []

        for mem in memories:
            meta = mem.get("metadata") or {}
            imp = mem.get("importance", 5)

            created = mem.get("created_at") or mem.get("last_accessed")
            if created:
                try:
                    t = datetime.fromisoformat(created)
                    if t.tzinfo is None:
                        t = t.replace(tzinfo=timezone.utc)
                    age_h = max(0.0, (now - t).total_seconds() / 3600.0)
                except (ValueError, TypeError):
                    age_h = 72.0
            else:
                age_h = 72.0

            age_factor = max(0.0, 1.0 - age_h / 168.0)
            arousal = meta.get("emotional_arousal", 0.0)
            importance = (imp if isinstance(imp, (int, float)) else 5) / 10.0
            confidence = meta.get("belief_confidence", 0.5)
            strength_str = meta.get("belief_strength", "uncertain")
            contradicted = 1.0 if strength_str == "contradicted" else 0.0

            priority = (
                age_factor * 0.3
                + arousal * 0.25
                + importance * 0.25
                + (1.0 - confidence) * 0.1
                + contradicted * 0.1
            )
            scored.append((round(priority, 4), mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:max_count]]

    def replay_memory(
        self,
        memory: Dict[str, Any],
        related_memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Re-evaluate a memory against related memories.

        Uses Chinese-aware tokenization (jieba if available)
        instead of naive whitespace split.
        """
        mid = memory.get("id", "")
        text = memory.get("text", "")

        suggestions: Dict[str, Any] = {
            "memory_id": mid,
            "confidence_adjustment": 0.0,
            "importance_adjustment": 0,
            "belief_update": None,
            "issues": [],
        }

        # Use Chinese-aware tokenization for overlap detection
        words_a = set(tokenize(text))

        support_count = 0
        conflict_count = 0
        for rel in related_memories:
            rel_text = (rel.get("text", "") or "")
            if not rel_text:
                continue
            # Chinese-aware tokenization
            words_b = set(tokenize(rel_text))
            overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
            if overlap > 0.2:
                # Check sentiment alignment via improved negation detection
                has_neg_a = _has_negation(text)
                has_neg_b = _has_negation(rel_text)
                if has_neg_a != has_neg_b:
                    conflict_count += 1
                else:
                    support_count += 1

        total = support_count + conflict_count
        if total > 0:
            ratio = support_count / total
            if ratio > 0.7:
                suggestions["confidence_adjustment"] = 0.1
                suggestions["issues"].append(f"被 {support_count} 条记忆支持")
            elif ratio < 0.3:
                suggestions["confidence_adjustment"] = -0.15
                suggestions["issues"].append(f"被 {conflict_count} 条记忆质疑")
                suggestions["belief_update"] = "needs_review"

        if len(related_memories) > 3:
            suggestions["importance_adjustment"] = 1

        return suggestions

    # ==================================================================
    # 2. Conversation Compression (LLM-driven)
    # ==================================================================

    def compress_conversations(
        self,
        conversations: List[Dict[str, str]],
        fingerprint: Optional[Dict[str, Any]] = None,
        max_summary_tokens: int = 500,
    ) -> Optional[Dict[str, Any]]:
        """
        Compress a batch of conversations into a structured summary.
        Uses keyword extraction + template (no LLM) to save tokens.
        Falls back to LLM only if heuristic result is insufficient.
        """
        if not conversations:
            return None

        # Default: use heuristic compression (saves LLM tokens)
        result = self._fallback_compress(conversations)
        if len(conversations) <= 2 and len(result.get("key_decisions", [])) == 0:
            # Very short conversations with no decisions: try LLM for deeper analysis
            fp = fingerprint or {}
            llm_result = self.llm.consolidate(conversations, fp, max_summary_tokens)
            if llm_result:
                llm_result["created_at"] = datetime.now(timezone.utc).isoformat()
                llm_result["original_turn_count"] = len(conversations)
                return llm_result
        return result

    @staticmethod
    def _fallback_compress(conversations: List[Dict[str, str]]) -> Dict[str, Any]:
        """Simple fallback compression without LLM. Uses keyword extraction for topics/entities."""
        all_topics: set[str] = set()
        all_entities: set[str] = set()
        summaries: list[str] = []
        key_decisions: list[str] = []
        preferences: list[str] = []
        emotional_moments: list[Dict[str, Any]] = []

        tech_terms = [
            "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C++", "C#",
            "Docker", "Kubernetes", "K8s", "Git", "GitHub", "GitLab",
            "React", "Vue", "Angular", "Node.js", "Django", "Flask",
            "MySQL", "PostgreSQL", "MongoDB", "Redis",
            "AWS", "Azure", "GCP", "Linux", "Windows", "MacOS",
            "VS Code", "Vim", "Emacs", "IntelliJ", "PyCharm",
        ]

        topic_keywords = {
            "编程": ["代码", "编程", "开发", "程序", "函数", "bug", "error"],
            "运维": ["部署", "服务器", "docker", "k8s", "运维", "监控"],
            "学习": ["学习", "教程", "文档", "怎么", "如何", "为什么"],
            "项目": ["项目", "任务", "需求", "功能", "版本", "发布"],
            "工具": ["工具", "软件", "插件", "扩展", "配置"],
            "生活": ["吃饭", "睡觉", "运动", "旅行", "购物", "电影", "音乐", "游戏"],
            "工作": ["工作", "会议", "面试", "薪水", "老板", "同事", "加班"],
            "技术": ["算法", "架构", "设计", "测试", "部署", "数据库", "API", "前端", "后端"],
        }

        decision_markers = ["决定", "选择", "确认", "用", "采用", "就用这个"]
        pref_markers = ["喜欢", "偏好", "习惯", "常用", "推荐"]

        for turn in conversations:
            user = turn.get("user", "")
            assistant = turn.get("assistant", "")

            if len(user) > 10:
                summaries.append(f"用户问: {user[:80]}")

            combined_text = user + " " + assistant

            for term in tech_terms:
                if term.lower() in combined_text.lower():
                    all_entities.add(term)

            matched_topic = False
            for topic, keywords in topic_keywords.items():
                for kw in keywords:
                    if kw in combined_text.lower():
                        all_topics.add(topic)
                        matched_topic = True
                        break

            for marker in decision_markers:
                if marker in user:
                    sentences = user.split("。")
                    for sent in sentences:
                        if marker in sent and len(key_decisions) < 5:
                            key_decisions.append(sent.strip()[:100])
                            break

            for marker in pref_markers:
                if marker in user:
                    sentences = user.split("。")
                    for sent in sentences:
                        if marker in sent and len(preferences) < 5:
                            preferences.append(sent.strip()[:100])
                            break

            # Extract emotional moments from user input
            for pattern, intensity, is_positive in _EMOTION_PATTERNS:
                match = pattern.search(user)
                if match:
                    if is_positive is True:
                        val = 0.8 if intensity == "high" else 0.6
                    elif is_positive is False:
                        val = -0.8 if intensity == "high" else -0.6
                    else:
                        val = 0.5  # neutral intensity marker (e.g. repeated punctuation)
                    emotional_moments.append({
                        "context": user[:80],
                        "emotion": match.group(0),
                        "intensity": val,
                    })
                    break

        # Add general topic only if no topics were matched at all
        if not all_topics:
            all_topics.add("general")

        return {
            "summary": "; ".join(summaries[:10]) if summaries else "无对话摘要",
            "key_decisions": key_decisions[:5],
            "preferences_revealed": preferences[:5],
            "emotional_moments": emotional_moments[:5],
            "entities_mentioned": list(all_entities)[:10],
            "topics": list(all_topics)[:5],
            "thinking_patterns": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "original_turn_count": len(conversations),
            "method": "fallback",
        }

    # ==================================================================
    # 3. Topic Timeline Building
    # ==================================================================

    def build_timeline_entries(
        self,
        compressed_summaries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extract timeline entries from compressed summaries.
        """
        entries = []
        for summary in compressed_summaries:
            # Get from LLM result
            tl_entry = summary.get("timeline_entry")
            if tl_entry and tl_entry.get("topic"):
                entries.append(tl_entry)
            else:
                # Fallback: use topics from summary
                for topic in summary.get("topics", [])[:2]:
                    entries.append({
                        "topic": topic,
                        "date": summary.get("created_at", "")[:10],
                        "summary": summary.get("summary", "")[:100],
                        "importance": 5,
                    })
        return entries

    # ==================================================================
    # 4. Cognitive Fingerprint Update
    # ==================================================================

    def update_fingerprint(
        self,
        current_fingerprint: Dict[str, Any],
        new_summaries: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Update cognitive fingerprint based on new conversation summaries.
        Uses LLM for deep analysis.
        """
        if not new_summaries:
            return None

        result = self.llm.update_fingerprint(current_fingerprint, new_summaries)
        if result:
            result["last_updated"] = datetime.now(timezone.utc).isoformat()
            return result

        # Fallback: extract patterns from summaries
        return self._fallback_fingerprint_update(current_fingerprint, new_summaries)

    @staticmethod
    def _fallback_fingerprint_update(
        current: Dict[str, Any],
        summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge summary data into fingerprint without LLM."""
        merged = dict(current)
        for s in summaries:
            for pattern in s.get("thinking_patterns", []):
                if pattern not in merged.get("thinking_patterns", []):
                    merged.setdefault("thinking_patterns", []).append(pattern)
            for pref in s.get("preferences_revealed", []):
                if pref not in merged.get("core_preferences", []):
                    merged.setdefault("core_preferences", []).append(pref)
        merged["last_updated"] = datetime.now(timezone.utc).isoformat()
        merged["version"] = merged.get("version", 0) + 1
        return merged

    # ==================================================================
    # 5. Relation Discovery
    # ==================================================================

    def set_embed_cache(self, cache: EmbeddingCache) -> None:
        """Inject an embedding cache for clustering."""
        self.embed_cache = cache

    def discover_relations(
        self,
        memories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Discover relations between memories.

        Uses embedding clustering when available, falls back to
        shared entity/topic pairwise comparison.
        """
        if self.embed_cache and memories:
            return self._embedding_discover_relations(memories)
        return self._heuristic_discover_relations(memories)

    def _embedding_discover_relations(
        self,
        memories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Discover relations via embedding clustering."""
        relations = []
        clusters = self.embed_cache.cluster_by_similarity(
            memories, threshold=0.5
        )
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            # All items in a cluster are related
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    a, b = cluster[i], cluster[j]
                    relations.append({
                        "source": a.get("id", ""),
                        "target": b.get("id", ""),
                        "relation_type": "semantic_cluster",
                        "confidence": 0.7,
                        "shared": [],
                    })
        return relations[:100]  # cap

    def _heuristic_discover_relations(
        self,
        memories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Discover relations via shared entities/topics (fallback)."""
        relations = []
        n = len(memories)
        max_pairs = 5000
        pairs_checked = 0

        for i in range(n):
            for j in range(i + 1, min(n, i + 20)):
                if pairs_checked >= max_pairs:
                    logger.warning(
                        "Heuristic relation discovery exceeded max_pairs=%d at i=%d, stopping",
                        max_pairs, i,
                    )
                    break
                pairs_checked += 1
                a, b = memories[i], memories[j]
                meta_a = a.get("metadata") or {}
                meta_b = b.get("metadata") or {}

                entities_a = set(str(e).lower() for e in meta_a.get("entities", []))
                entities_b = set(str(e).lower() for e in meta_b.get("entities", []))
                topics_a = set(str(t).lower() for t in meta_a.get("topics", []))
                topics_b = set(str(t).lower() for t in meta_b.get("topics", []))

                shared_entities = entities_a & entities_b
                shared_topics = topics_a & topics_b

                if shared_entities or shared_topics:
                    confidence = min(
                        1.0,
                        0.4 + len(shared_entities) * 0.25 + len(shared_topics) * 0.15,
                    )
                    if confidence > 0.5:
                        rel_type = "co_occurs" if shared_entities else "same_topic"
                        relations.append({
                            "source": a.get("id", ""),
                            "target": b.get("id", ""),
                            "relation_type": rel_type,
                            "confidence": round(confidence, 3),
                            "shared": list(shared_entities | shared_topics)[:5],
                        })
            if pairs_checked >= max_pairs:
                break

        return relations

    def apply_relations(
        self,
        relations: List[Dict[str, Any]],
        gm_record_func: Callable[..., Any],
    ) -> int:
        """Create relation edges via gm_record. Returns count created."""
        created = 0
        for rel in relations:
            try:
                gm_record_func(
                    source=rel.get("source", ""),
                    target=rel.get("target", ""),
                    relation=rel.get("relation_type", "related"),
                    context=json.dumps(rel.get("shared", []), ensure_ascii=False),
                )
                created += 1
            except Exception as e:
                logger.debug("Graph record failed: %s", e)
        return created


# ======================================================================
# Negation detection helper
# ======================================================================

# Negation words that truly negate the sentence meaning
_NEGATION_PATTERNS = [
    re.compile(r'(?<![好不])不(?=[好是行对喜欢讨厌想要需要应该可以能会])'),
    re.compile(r'没有?(?:做|去|来|看到|听到|知道)'),
    re.compile(r'无(?=[法效意义用关])'),
    re.compile(r'非(?=[法])'),
    re.compile(r'未(?=[完成知])'),
    re.compile(r'(?<![毫])无(?=[疑问问题])'),
    re.compile(r'没办?法'),
]


def _has_negation(text: str) -> bool:
    """Check if text contains true negation (Chinese-aware).

    Avoids false positives like "不错" (not bad = good), "没问题" (no problem = ok),
    "非常好" (very good).
    """
    # First check explicit negation patterns
    for pattern in _NEGATION_PATTERNS:
        if pattern.search(text):
            return True
    # Fallback: check standalone negation words with context
    # Exclude false positives: 不好, 不错, 不行, 不是, 不用, 不太, 不了, 不起, 不住, 不到, 不过
    standalone_neg = re.findall(r'(?<![好不])不(?!好|错|行|是|用|太|了|起|住|到|过)', text)
    if standalone_neg:
        return True
    return False


# ======================================================================
# Self-test
# ======================================================================


def _self_test():
    """Run: python -m hms.scripts.consolidation"""
    engine = ConsolidationEngine()

    # Test replay selection
    memories = [
        {"id": "m1", "importance": 8, "metadata": {"emotional_arousal": 0.7}},
        {"id": "m2", "importance": 3, "metadata": {"emotional_arousal": 0.1}},
        {"id": "m3", "importance": 6, "metadata": {"belief_strength": "contradicted"}},
    ]
    selected = engine.select_for_replay(memories, max_count=2)
    assert len(selected) == 2
    print(f"[replay] selected {[m['id'] for m in selected]}")

    # Test fallback compression
    conversations = [
        {"user": "Python怎么安装", "assistant": "去官网下载"},
        {"user": "Docker怎么用", "assistant": "先安装Docker"},
    ]
    result = engine.compress_conversations(conversations)
    assert result is not None
    assert result["original_turn_count"] == 2
    print(f"[compress] summary={result['summary'][:50]}...")

    # Test relation discovery
    mems = [
        {"id": "a", "metadata": {"entities": ["Python"], "topics": ["编程"]}},
        {"id": "b", "metadata": {"entities": ["Python"], "topics": ["AI"]}},
    ]
    rels = engine.discover_relations(mems)
    assert len(rels) >= 1
    print(f"[relations] found {len(rels)} relations")

    # Test negation detection
    assert _has_negation("我不喜欢这个")
    assert not _has_negation("这个不错")
    print("[negation] OK")

    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
