"""
StorageAdapter v5 — SQLite + Ollama Embedding Vector Storage.

Replaces LanceDB with SQLite + Ollama qwen3-embedding:0.6b for
semantic vector search via cosine similarity.

Key changes from v4:
  - No LanceDB dependency
  - Vectors stored as JSON blob in SQLite BLOB column
  - Ollama embedding API for vector generation
  - FTS5 for keyword search + cosine similarity reranking
"""

from __future__ import annotations
import json
import math
import os
import sqlite3
import threading
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hms.adapter")

# Optional Ollama API
try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ─────────────────────────────────────────────────────────────────────────────
# Ollama Embedding
# ─────────────────────────────────────────────────────────────────────────────

def _get_embedding(texts: List[str], config: Dict[str, Any]) -> List[List[float]]:
    """
    Call Ollama OpenAI-compatible /v1/embeddings API.
    Falls back to char-ngram if Ollama is unavailable.

    Config keys:
      - embedding_ollama_base_url: from config or HMS_OLLAMA_BASE_URL env var
      - embedding_ollama_model: e.g. "qwen3-embedding:0.6b"
      - embedding_ollama_dim: expected vector dimension (default 1024)
    """
    if not _HAS_REQUESTS:
        logger.warning("requests not available, using char-ngram fallback")
        return [_char_ngram_embedding(t, dim=config.get("embedding_ollama_dim", 1024)) for t in texts]

    base_url = config.get("embedding_ollama_base_url", os.environ.get("HMS_OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    model = config.get("embedding_ollama_model", "qwen3-embedding:0.6b")
    dim = config.get("embedding_ollama_dim", 1024)

    try:
        resp = requests.post(
            f"{base_url}/v1/embeddings",
            json={"model": model, "input": texts},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"])
        return [item["embedding"] for item in data]
    except Exception as e:
        logger.warning("Ollama embedding failed: %s, falling back to char-ngram", e)
        return [_char_ngram_embedding(t, dim=dim) for t in texts]


def _char_ngram_embedding(text: str, dim: int = 1024) -> List[float]:
    """Pure-Python char n-gram embedding fallback (no external deps)."""
    import hashlib
    text = text.lower().strip()
    if not text:
        return [0.0] * dim
    vec = [0.0] * dim
    for n in (2, 3):
        for i in range(len(text) - n + 1):
            g = text[i:i + n]
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16) % dim
            vec[h] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────────────────────────────────────
# StorageAdapter
# ─────────────────────────────────────────────────────────────────────────────

class StorageAdapter:
    """
    SQLite storage with Ollama embedding vectors.

    Schema:
      memories(id INTEGER PRIMARY KEY, text TEXT, vector_json TEXT,
               category TEXT, importance INTEGER, metadata TEXT,
               created_at TEXT, updated_at TEXT, access_count INTEGER)

    - vector_json: JSON-serialized list of floats (stored as TEXT in SQLite)
    - FTS5 virtual table for keyword search
    - Cosine similarity reranking on FTS results
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        data_dir = config.get("cache_dir", config.get("dataDir", "data"))
        os.makedirs(data_dir, exist_ok=True)
        self._db_path = os.path.join(data_dir, "memories.db")
        self._lock = threading.Lock()
        self._embedding_dim = config.get("embedding_ollama_dim", 1024)
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            conn = self._conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    vector_json TEXT DEFAULT '[]',
                    category TEXT DEFAULT 'fact',
                    importance INTEGER DEFAULT 5,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now')),
                    access_count INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_mem_imp ON memories(importance);
                CREATE INDEX IF NOT EXISTS idx_mem_cat ON memories(category);
                CREATE INDEX IF NOT EXISTS idx_mem_created ON memories(created_at);

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    text, category, metadata,
                    content='memories', content_rowid='id'
                );

                CREATE TRIGGER IF NOT EXISTS trg_mem_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, text, category, metadata)
                    VALUES (new.id, new.text, new.category, new.metadata);
                END;
                CREATE TRIGGER IF NOT EXISTS trg_mem_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, text, category, metadata)
                    VALUES ('delete', old.id, old.text, old.category, old.metadata);
                END;
                CREATE TRIGGER IF NOT EXISTS trg_mem_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, text, category, metadata)
                    VALUES ('delete', old.id, old.text, old.category, old.metadata);
                    INSERT INTO memories_fts(rowid, text, category, metadata)
                    VALUES (new.id, new.text, new.category, new.metadata);
                END;
            """)
            conn.close()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def store(self, text: str, category: str = "fact", importance: int = 5,
              metadata: str = "{}", vector: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Store a memory with its embedding vector.

        If vector is not provided, it will be generated via Ollama.
        """
        if vector is None:
            vector = _get_embedding([text], self.cfg)[0]

        vector_json = json.dumps(vector)

        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute(
                    "INSERT INTO memories (text, vector_json, category, importance, metadata) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (text, vector_json, category, importance, metadata),
                )
                conn.commit()
                return {"id": cur.lastrowid, "status": "stored", "vector_dim": len(vector)}
            finally:
                conn.close()

    def store_with_dedup(self, text: str, category: str = "fact", importance: int = 5,
                         metadata: str = "{}", embed_cache=None) -> Dict[str, Any]:
        """
        Store with deduplication using embedding similarity.
        Falls back to text-level dedup if embed_cache is unavailable.
        """
        threshold = self.cfg.get("dedupSimilarityThreshold", 0.95)
        candidates = self.recall(query=text, top_k=5)

        if embed_cache:
            try:
                for mem in candidates:
                    existing_text = mem.get("text", "")
                    if existing_text:
                        sim = embed_cache.similarity(text, existing_text)
                        if sim >= threshold:
                            new_imp = max(mem.get("importance", 0), importance)
                            self.update(str(mem["id"]), importance=new_imp)
                            return {"id": mem["id"], "status": "merged", "similarity": sim}
            except Exception as e:
                logger.warning("Dedup similarity check failed: %s", e)

        return self.store(text, category, importance, metadata)

    def recall(self, query: str, top_k: int = 5,
               category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Semantic recall using Ollama embeddings + cosine similarity.

        1. Generate query vector via Ollama
        2. Get candidate memories (FTS5 keyword match OR all if no query)
        3. Compute cosine similarity and return top-k
        """
        if not query:
            with self._lock:
                conn = self._conn()
                try:
                    if category:
                        rows = conn.execute(
                            "SELECT * FROM memories WHERE category=? ORDER BY created_at DESC LIMIT ?",
                            (category, top_k),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
                            (top_k,),
                        ).fetchall()
                    return [dict(r) for r in rows]
                finally:
                    conn.close()

        # Generate query embedding
        query_vec = _get_embedding([query], self.cfg)[0]

        with self._lock:
            conn = self._conn()
            try:
                # FTS5 keyword search to get candidates
                if category:
                    rows = conn.execute(
                        """SELECT m.*, rank FROM memories m
                           JOIN memories_fts fts ON m.id = fts.rowid
                           WHERE memories_fts MATCH ? AND m.category = ?
                           ORDER BY rank LIMIT ?""",
                        (query, category, top_k * 3),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT m.*, rank FROM memories m
                           JOIN memories_fts fts ON m.id = fts.rowid
                           WHERE memories_fts MATCH ?
                           ORDER BY rank LIMIT ?""",
                        (query, top_k * 3),
                    ).fetchall()

                # If FTS returned nothing, fall back to all memories
                if not rows:
                    rows = conn.execute(
                        "SELECT * FROM memories LIMIT ?",
                        (top_k * 3,),
                    ).fetchall()

                # Cosine similarity reranking
                scored: List[tuple] = []
                for row in rows:
                    try:
                        vec = json.loads(row["vector_json"])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
                    sim = _cosine_similarity(query_vec, vec)
                    scored.append((sim, dict(row)))

                scored.sort(key=lambda x: x[0], reverse=True)
                return [mem for _, mem in scored[:top_k]]
            finally:
                conn.close()

    def forget(self, memory_id: str) -> Dict[str, Any]:
        with self._lock:
            conn = self._conn()
            try:
                conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                conn.commit()
                return {"status": "forgotten", "id": memory_id}
            finally:
                conn.close()

    def update(self, memory_id: str, **kwargs) -> Dict[str, Any]:
        sets, vals = [], []
        for k, v in kwargs.items():
            if k in ("importance", "category", "metadata", "text"):
                sets.append(f"{k} = ?")
                vals.append(v)
        if not sets:
            return {"status": "no_changes"}
        sets.append("updated_at = datetime('now')")
        vals.append(memory_id)
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", vals)
                conn.commit()
                return {"status": "updated", "id": memory_id}
            finally:
                conn.close()

    def increment_access(self, memory_id: str) -> None:
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                    (memory_id,),
                )
                conn.commit()
            finally:
                conn.close()

    def get_vector(self, memory_id: str) -> Optional[List[float]]:
        with self._lock:
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT vector_json FROM memories WHERE id = ?", (memory_id,)
                ).fetchone()
                if row:
                    return json.loads(row["vector_json"])
                return None
            finally:
                conn.close()

    def health_check(self) -> Dict[str, Any]:
        try:
            with self._lock:
                conn = self._conn()
                try:
                    row = conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
                    total = row["cnt"]
                    cat_rows = conn.execute(
                        "SELECT category, COUNT(*) as cnt FROM memories GROUP BY category"
                    ).fetchall()
                    categories = {r["category"]: r["cnt"] for r in cat_rows}
                    return {
                        "status": "ok",
                        "total_memories": total,
                        "categories": categories,
                        "embedding_backend": "ollama",
                        "embedding_dim": self._embedding_dim,
                    }
                finally:
                    conn.close()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def close(self) -> None:
        pass

    def get_all_memories(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all memories for consolidation/forgetting passes."""
        with self._lock:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()
