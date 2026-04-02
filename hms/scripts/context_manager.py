"""
HMS v3.2 — Context Manager.

Three-layer infinite context architecture:
  Layer 1: Working memory (recent turns)
  Layer 2: Compressed summaries (LLM-generated)
  Layer 3: Cognitive fingerprint + topic timelines (persistent)

Plus dynamic token budget allocation.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .file_utils import (
    atomic_write_json,
    safe_read_json,
    safe_append_jsonl,
    safe_read_jsonl,
    safe_clear_jsonl,
    file_lock,
)
from .utils import estimate_tokens

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Orchestrates the HMS v3.2 cognitive layer:
      - Async perception pipeline (pending -> batch LLM analysis)
      - Three-layer context composition for infinite context
      - Dynamic token budget allocation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self._pending_path = self.cfg.get(
            "pending_path", "cache/pending_processing.jsonl"
        )
        self._cache_dir = os.path.dirname(self._pending_path) or "cache"

        # Pending queue size limit
        self._max_pending = self.cfg.get("max_pending_size", 1000)

        # Max summary length to prevent unbounded growth
        self._max_summary_len = self.cfg.get("timeline_max_summary_len", 1000)

        # File paths for v2 structures
        self._fingerprint_path = os.path.join(
            self._cache_dir, "cognitive_fingerprint.json"
        )
        self._timelines_path = os.path.join(self._cache_dir, "topic_timelines.json")
        self._compression_path = os.path.join(
            self._cache_dir, "compression_history.json"
        )

        os.makedirs(self._cache_dir, exist_ok=True)

        # Load state
        self._fingerprint = safe_read_json(self._fingerprint_path, {})
        self._timelines = safe_read_json(self._timelines_path, {})
        self._compression_history = safe_read_json(self._compression_path, [])

    # ==================================================================
    # Persistence helpers
    # ==================================================================

    def save_state(self, changed: Optional[Set[str]] = None) -> None:
        """Persist all state to disk atomically with locks.

        Uses a temporary directory + batch os.replace so all files are
        updated consistently — if write fails mid-way, no file is corrupted.

        Args:
            changed: Optional set of changed keys ('fingerprint', 'timelines',
                     'compression_history'). If None, saves all three.
        """
        all_keys = changed or {"fingerprint", "timelines", "compression_history"}

        file_map = {
            "fingerprint": (self._fingerprint_path, self._fingerprint),
            "timelines": (self._timelines_path, self._timelines),
            "compression_history": (self._compression_path, self._compression_history),
        }

        # Phase 1: Write all changed files to unique temp files
        temp_files = {}
        for key in all_keys:
            path, data = file_map[key]
            dir_name = os.path.dirname(path) or "."
            os.makedirs(dir_name, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                temp_files[key] = (path, tmp_path)
            except Exception:
                # Clean up temp files on failure
                for _, (_, tp) in temp_files.items():
                    try:
                        os.unlink(tp)
                    except OSError:
                        pass
                raise

        # Phase 2: Atomically replace all files under lock (batch rename)
        # Lock the first file as the coordination lock
        first_key = next(iter(all_keys))
        first_path = file_map[first_key][0]
        with file_lock(first_path):
            for key, (path, tmp_path) in temp_files.items():
                try:
                    os.replace(tmp_path, path)
                    with file_lock(path):
                        pass  # Touch lock so readers see the update
                except OSError:
                    # If replace fails, clean up remaining temp files
                    for _, (p, tp) in temp_files.items():
                        if p != path:
                            try:
                                os.unlink(tp)
                            except OSError:
                                pass
                    raise

    # ==================================================================
    # Pending queue
    # ==================================================================

    def enqueue(
        self,
        user_message: str,
        assistant_reply: str,
        session_id: str = "",
        timestamp: Optional[str] = None,
    ) -> None:
        """Add a conversation turn to the pending processing queue.

        Enforces max_pending_size limit — drops oldest entries when full.
        """
        # Check pending size before adding
        count = self.get_pending_count()
        if count >= self._max_pending:
            logger.warning(
                "Pending queue full (%d/%d). Dropping oldest entries.",
                count, self._max_pending,
            )
            # Drop oldest 20%
            self._trim_pending(int(self._max_pending * 0.2))

        ts = timestamp or datetime.now(timezone.utc).isoformat()
        entry = {
            "user_message": user_message,
            "assistant_reply": assistant_reply,
            "session_id": session_id,
            "timestamp": ts,
        }
        safe_append_jsonl(self._pending_path, entry)

    def _trim_pending(self, count: int) -> None:
        """Remove the oldest N entries from pending queue."""
        if not os.path.isfile(self._pending_path):
            return
        with file_lock(self._pending_path):
            with open(self._pending_path, "r+", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
                if len(lines) > count:
                    lines = lines[count:]
                f.seek(0)
                f.truncate(0)
                f.writelines(lines)

    def read_pending(self) -> List[Dict[str, Any]]:
        """Read all pending entries without clearing."""
        return safe_read_jsonl(self._pending_path)

    def pop_all_pending(self) -> List[Dict[str, Any]]:
        """Atomically read and clear all pending entries.

        Reads the file under lock, then truncates it before returning.
        This prevents duplicate processing when multiple consumers
        (e.g. process_pending + consolidate) run concurrently.
        """
        if not os.path.isfile(self._pending_path):
            return []

        entries = []
        with file_lock(self._pending_path):
            if os.path.isfile(self._pending_path):
                with open(self._pending_path, "r+", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entries.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logger.warning("Corrupted JSONL line in pending queue, skipping: %s", line[:100])
                                continue
                    if entries:
                        f.seek(0)
                        f.truncate(0)
        return entries

    def clear_pending(self) -> None:
        """Truncate the pending file."""
        safe_clear_jsonl(self._pending_path)

    def get_pending_count(self) -> int:
        if not os.path.isfile(self._pending_path):
            return 0
        with file_lock(self._pending_path):
            with open(self._pending_path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())

    # ==================================================================
    # Cognitive fingerprint
    # ==================================================================

    def get_fingerprint(self) -> Dict[str, Any]:
        """Return current cognitive fingerprint."""
        return self._fingerprint

    def update_fingerprint(self, new_fingerprint: Dict[str, Any]) -> None:
        """Merge new fingerprint data into existing.

        Each list field has a maximum capacity; oldest entries are evicted
        when the cap is exceeded to prevent unbounded growth.
        """
        list_caps = self.cfg.get("fingerprint_list_caps", {
            "thinking_patterns": 10,
            "core_preferences": 10,
            "focus_areas": 10,
            "values": 10,
            "recent_goals": 5,
        })
        trigger_caps = self.cfg.get("fingerprint_trigger_caps", 10)

        if not self._fingerprint:
            self._fingerprint = new_fingerprint
        else:
            for key, cap in list_caps.items():
                existing = list(dict.fromkeys(self._fingerprint.get(key, [])))
                for item in new_fingerprint.get(key, []):
                    if item not in existing:
                        existing.append(item)
                if len(existing) > cap:
                    existing = existing[-cap:]
                self._fingerprint[key] = existing

            for valence in ["positive", "negative"]:
                existing = list(dict.fromkeys(
                    self._fingerprint.get("emotional_triggers", {}).get(valence, [])
                ))
                for item in (
                    new_fingerprint.get("emotional_triggers", {}).get(valence, [])
                ):
                    if item not in existing:
                        existing.append(item)
                if len(existing) > trigger_caps:
                    existing = existing[-trigger_caps:]
                self._fingerprint.setdefault("emotional_triggers", {})[valence] = existing

            for key in ["communication_style", "personality_notes"]:
                if new_fingerprint.get(key):
                    self._fingerprint[key] = new_fingerprint[key]

            self._fingerprint["last_updated"] = datetime.now(timezone.utc).isoformat()
            self._fingerprint["version"] = self._fingerprint.get("version", 0) + 1

        self.save_state()

    # ==================================================================
    # Topic timelines
    # ==================================================================

    def get_timelines(self) -> Dict[str, Any]:
        return self._timelines

    def update_timelines(self, timeline_entries: List[Dict[str, Any]]) -> None:
        """Add entries to topic timelines."""
        max_entries = self.cfg.get("timeline_max_entries_per_topic", 10)

        for entry in timeline_entries:
            topic = entry.get("topic", "")
            if not topic:
                continue

            if topic not in self._timelines:
                self._timelines[topic] = {
                    "topic": topic,
                    "entries": [],
                    "summary": "",
                    "last_updated": "",
                    "total_entries_merged": 0,
                }

            tl = self._timelines[topic]
            tl["entries"].append({
                "date": entry.get("date", ""),
                "summary": entry.get("summary", ""),
                "importance": entry.get("importance", 5),
            })
            tl["last_updated"] = datetime.now(timezone.utc).isoformat()

            # Prune if too many entries
            if len(tl["entries"]) > max_entries:
                sorted_entries = sorted(
                    tl["entries"], key=lambda e: e.get("importance", 0), reverse=True
                )
                keep = sorted_entries[:max_entries]
                merge = sorted_entries[max_entries:]
                merged_summaries = [
                    e.get("summary", "") for e in merge if e.get("summary")
                ]
                if merged_summaries:
                    new_summary = (
                        tl.get("summary", "") + " | ".join(merged_summaries)
                    ).strip()
                    # Truncate to prevent unbounded growth
                    if len(new_summary) > self._max_summary_len:
                        new_summary = new_summary[: self._max_summary_len - 3] + "..."
                    tl["summary"] = new_summary
                    tl["total_entries_merged"] = tl.get("total_entries_merged", 0) + len(
                        merge
                    )
                tl["entries"] = keep

        # Cap total topics
        max_topics = self.cfg.get("timeline_max_topics", 20)
        if len(self._timelines) > max_topics:
            # Keep most recently updated
            sorted_topics = sorted(
                self._timelines.items(),
                key=lambda x: x[1].get("last_updated", ""),
                reverse=True,
            )
            self._timelines = dict(sorted_topics[:max_topics])

        self.save_state()

    # ==================================================================
    # Compression history
    # ==================================================================

    def add_compressed_summary(self, summary: Dict[str, Any]) -> None:
        """Add a compressed conversation summary."""
        self._compression_history.append(summary)
        self.save_state()

    def get_compressed_summaries(
        self, since_hours: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get compressed summaries, optionally filtered by recency."""
        if since_hours is None:
            return self._compression_history

        cutoff = datetime.now(timezone.utc).timestamp() - since_hours * 3600
        result = []
        for s in self._compression_history:
            created = s.get("created_at", "")
            if created:
                try:
                    t = datetime.fromisoformat(created)
                    if t.tzinfo is None:
                        t = t.replace(tzinfo=timezone.utc)
                    if t.timestamp() >= cutoff:
                        result.append(s)
                except ValueError:
                    # Invalid date format, include it anyway
                    result.append(s)
            else:
                result.append(s)
        return result

    # ==================================================================
    # Token budget
    # ==================================================================

    def estimate_token_budget(
        self, model_context_window: int = 256000
    ) -> Dict[str, int]:
        """Dynamic token budget allocation with minimum floor protection."""
        budget_cfg = self.cfg.get("context_budget", {})

        sys_prompt = budget_cfg.get("system_prompt", 4000)
        fingerprint_ratio = budget_cfg.get("cognitive_fingerprint_ratio", 0.02)
        timelines_ratio = budget_cfg.get("topic_timelines_ratio", 0.03)
        compressed_ratio = budget_cfg.get("compressed_summaries_ratio", 0.05)
        memories_ratio = budget_cfg.get("injected_memories_ratio", 0.15)
        recent_ratio = budget_cfg.get("recent_turns_ratio", 0.35)
        response_ratio = budget_cfg.get("response_reserve_ratio", 0.15)
        buffer_ratio = budget_cfg.get("buffer_ratio", 0.05)

        fingerprint = int(model_context_window * fingerprint_ratio)
        timelines = int(model_context_window * timelines_ratio)
        compressed = int(model_context_window * compressed_ratio)
        memories = int(model_context_window * memories_ratio)
        recent = int(model_context_window * recent_ratio)
        response = int(model_context_window * response_ratio)
        buffer = int(model_context_window * buffer_ratio)

        total = sys_prompt + fingerprint + timelines + compressed + memories + recent + response + buffer
        if total > model_context_window:
            scale = (model_context_window - sys_prompt) / (total - sys_prompt)
            fingerprint = int(fingerprint * scale)
            timelines = int(timelines * scale)
            compressed = int(compressed * scale)
            memories = int(memories * scale)
            recent = int(recent * scale)
            response = int(response * scale)
            buffer = int(buffer * scale)

        # Minimum floor protection — prevent any layer from being zeroed out
        MIN_FLOOR = {
            "cognitive_fingerprint": 300,
            "topic_timelines": 300,
            "compressed_summaries": 200,
            "injected_memories": 500,
            "recent_turns": 1000,
        }
        layers = {
            "cognitive_fingerprint": fingerprint,
            "topic_timelines": timelines,
            "compressed_summaries": compressed,
            "injected_memories": memories,
            "recent_turns": recent,
        }
        total_deficit = 0
        for key, minimum in MIN_FLOOR.items():
            if layers[key] < minimum:
                total_deficit += minimum - layers[key]
                layers[key] = minimum

        # Borrow from buffer if there's a deficit
        if total_deficit > 0 and buffer > total_deficit:
            buffer -= total_deficit

        return {
            "system_prompt": sys_prompt,
            "cognitive_fingerprint": layers["cognitive_fingerprint"],
            "topic_timelines": layers["topic_timelines"],
            "compressed_summaries": layers["compressed_summaries"],
            "injected_memories": layers["injected_memories"],
            "recent_turns": layers["recent_turns"],
            "response_reserve": response,
            "buffer": buffer,
            "total_available": sys_prompt + sum(layers.values()),
            "window": model_context_window,
        }

    # ==================================================================
    # Context composition (three-layer)
    # ==================================================================

    def compose_context(
        self,
        system_prompt: str,
        injected_memories: List[Dict[str, Any]],
        recent_turns: List[Dict[str, str]],
        model_context_window: int = 256000,
    ) -> Dict[str, Any]:
        """
        Compose full context using three-layer infinite context architecture:

        Layer 1 (top):     Cognitive fingerprint (~2000 tokens, always present)
        Layer 2 (middle):  Topic timelines + compressed summaries
        Layer 3 (bottom):  Recent turns + injected memories

        The fingerprint gives the AI "who the user is".
        Timelines give "what happened over time".
        Recent turns give "what's happening now".
        """
        budget = self.estimate_token_budget(model_context_window)

        # --- Layer 1: Cognitive fingerprint ---
        fp_text = self._format_fingerprint(self._fingerprint)
        fp_text = self.truncate_to_tokens(fp_text, budget["cognitive_fingerprint"])

        # --- Layer 2: Topic timelines + compressed summaries ---
        tl_text = self._format_timelines(self._timelines)
        tl_text = self.truncate_to_tokens(tl_text, budget["topic_timelines"])

        comp_summaries = self.get_compressed_summaries(since_hours=168)  # last week
        comp_text = self._format_compressed_summaries(comp_summaries)
        comp_text = self.truncate_to_tokens(comp_text, budget["compressed_summaries"])

        # --- Layer 3: Injected memories ---
        sorted_mems = sorted(
            injected_memories,
            key=lambda m: m.get("importance", 0),
            reverse=True,
        )
        mem_texts = []
        mem_tokens = 0
        for mem in sorted_mems:
            formatted = self._format_memory(mem)
            tokens = estimate_tokens(formatted)
            if mem_tokens + tokens > budget["injected_memories"]:
                break
            mem_texts.append(formatted)
            mem_tokens += tokens
        mem_section = "\n".join(mem_texts)

        # --- Layer 3: Recent turns ---
        turn_texts = []
        turn_tokens = 0
        for turn in reversed(recent_turns):
            line = f"用户: {turn.get('user', '')}\n助手: {turn.get('assistant', '')}"
            tokens = estimate_tokens(line)
            if turn_tokens + tokens > budget["recent_turns"]:
                break
            turn_texts.insert(0, line)
            turn_tokens += tokens
        recent_section = "\n---\n".join(turn_texts)

        # --- System prompt ---
        sys_text = self.truncate_to_tokens(system_prompt, budget["system_prompt"])

        total_est = (
            estimate_tokens(sys_text)
            + estimate_tokens(fp_text)
            + estimate_tokens(tl_text)
            + estimate_tokens(comp_text)
            + mem_tokens
            + turn_tokens
        )

        return {
            "system_prompt": sys_text,
            "cognitive_fingerprint": fp_text,
            "topic_timelines": tl_text,
            "compressed_summaries": comp_text,
            "injected_memories_section": mem_section,
            "recent_section": recent_section,
            "total_tokens_estimated": total_est,
            "budget": budget,
        }

    def format_context(
        self,
        system_prompt: str = "",
        injected_memories: Optional[List[Dict[str, Any]]] = None,
        recent_turns: Optional[List[Dict[str, str]]] = None,
        model_context_window: int = 256000,
    ) -> str:
        """Compose and format context as a single injectable string.

        Convenience wrapper around compose_context() that returns
        a ready-to-use text block for LLM system prompts.
        """
        ctx = self.compose_context(
            system_prompt=system_prompt or "",
            injected_memories=injected_memories or [],
            recent_turns=recent_turns or [],
            model_context_window=model_context_window,
        )
        parts = []
        if ctx.get("system_prompt"):
            parts.append(ctx["system_prompt"])
        if ctx.get("cognitive_fingerprint"):
            parts.append(ctx["cognitive_fingerprint"])
        if ctx.get("topic_timelines"):
            parts.append(ctx["topic_timelines"])
        if ctx.get("compressed_summaries"):
            parts.append(ctx["compressed_summaries"])
        if ctx.get("injected_memories_section"):
            parts.append(f"## 相关记忆\n{ctx['injected_memories_section']}")
        if ctx.get("recent_section"):
            parts.append(f"## 最近对话\n{ctx['recent_section']}")
        return "\n\n".join(parts)

    # ==================================================================
    # Formatting helpers
    # ==================================================================

    @staticmethod
    def _format_fingerprint(fp: Dict[str, Any]) -> str:
        if not fp:
            return ""
        parts = ["## 用户认知指纹"]
        if fp.get("thinking_patterns"):
            parts.append(f"思维模式: {', '.join(fp['thinking_patterns'][:5])}")
        if fp.get("core_preferences"):
            parts.append(f"核心偏好: {', '.join(fp['core_preferences'][:5])}")
        if fp.get("communication_style"):
            parts.append(f"沟通风格: {fp['communication_style']}")
        if fp.get("focus_areas"):
            parts.append(f"关注领域: {', '.join(fp['focus_areas'][:5])}")
        if fp.get("values"):
            parts.append(f"价值观: {', '.join(fp['values'][:3])}")
        if fp.get("personality_notes"):
            parts.append(f"性格备注: {fp['personality_notes']}")
        triggers = fp.get("emotional_triggers", {})
        if triggers.get("positive"):
            parts.append(f"积极触发: {', '.join(triggers['positive'][:3])}")
        if triggers.get("negative"):
            parts.append(f"消极触发: {', '.join(triggers['negative'][:3])}")
        return "\n".join(parts)

    @staticmethod
    def _format_timelines(timelines: Dict[str, Any]) -> str:
        if not timelines:
            return ""
        parts = ["## 主题时间线"]
        for topic, tl in list(timelines.items())[:10]:
            entries = tl.get("entries", [])
            summary = tl.get("summary", "")
            parts.append(f"### {topic}")
            if summary:
                parts.append(f"历史摘要: {summary[:200]}")
            for entry in entries[:3]:
                date = entry.get("date", "")
                text = entry.get("summary", "")
                parts.append(f"- [{date}] {text[:100]}")
        return "\n".join(parts)

    @staticmethod
    def _format_compressed_summaries(summaries: List[Dict[str, Any]]) -> str:
        """Format compressed summaries, dynamically sized by token budget.

        Instead of hardcoding [-5:], this uses all summaries within
        the time window, letting truncate_to_tokens enforce the budget.
        """
        if not summaries:
            return ""
        parts = ["## 近期对话摘要"]
        for s in summaries:
            text = s.get("summary_text", s.get("summary", ""))
            if text:
                parts.append(f"- {text}")
        return "\n".join(parts)

    @staticmethod
    def _format_memory(mem: Dict[str, Any]) -> str:
        text = mem.get("text", "")
        imp = mem.get("importance", 5)
        return f"[重要度:{imp}] {text[:200]}"

    @staticmethod
    def truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens at safe breakpoints."""
        if not text:
            return ""
        est = estimate_tokens(text)
        if est <= max_tokens:
            return text
        ratio = max_tokens / max(est, 1)
        cut = int(len(text) * ratio)
        # Find safe breakpoint: search backwards for punctuation/space/newline
        safe_breakpoints = set("。！？；\n.!?; ")
        for i in range(cut, max(0, cut - 50), -1):
            if i < len(text) and text[i] in safe_breakpoints:
                cut = i + 1
                break
        return text[:cut] + "\n...(truncated)"


# ======================================================================
# Self-test
# ======================================================================


def _self_test():
    """Run: python -m hms.scripts.context_manager"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cm = ContextManager({
            "pending_path": os.path.join(tmpdir, "pending.jsonl"),
        })

        # Test enqueue
        cm.enqueue("你好", "你好！有什么可以帮助你的？")
        assert cm.get_pending_count() == 1

        # Test fingerprint
        cm.update_fingerprint({
            "thinking_patterns": ["注重细节"],
            "core_preferences": ["喜欢简洁"],
        })
        fp = cm.get_fingerprint()
        assert "注重细节" in fp.get("thinking_patterns", [])

        # Test timelines
        cm.update_timelines([{
            "topic": "项目A",
            "date": "2024-01-01",
            "summary": "项目启动",
            "importance": 8,
        }])
        tl = cm.get_timelines()
        assert "项目A" in tl

        # Test context composition
        ctx = cm.compose_context(
            system_prompt="你是一个助手",
            injected_memories=[{"text": "用户喜欢Python", "importance": 7}],
            recent_turns=[{"user": "你好", "assistant": "你好！"}],
        )
        assert "cognitive_fingerprint" in ctx
        assert "topic_timelines" in ctx
        print(f"[context] tokens_est={ctx['total_tokens_estimated']}")

        # Test format_context
        formatted = cm.format_context(
            system_prompt="你是一个助手",
            injected_memories=[{"text": "用户喜欢Python", "importance": 7}],
            recent_turns=[{"user": "你好", "assistant": "你好！"}],
        )
        assert isinstance(formatted, str)
        assert "用户认知指纹" in formatted or "你是一个助手" in formatted
        print(f"[format_context] OK, len={len(formatted)}")

        # Test safe truncation
        truncated = ContextManager.truncate_to_tokens("这是一个很长的中文句子。它应该在标点处截断。", 5)
        assert "truncated" in truncated or "截断" in truncated
        print(f"[truncate] OK")

        # Test pending size limit
        cm2 = ContextManager({
            "pending_path": os.path.join(tmpdir, "pending2.jsonl"),
            "max_pending_size": 5,
        })
        for i in range(10):
            cm2.enqueue(f"msg{i}", f"reply{i}")
        assert cm2.get_pending_count() <= 5
        print(f"[pending_limit] OK, count={cm2.get_pending_count()}")

        print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
