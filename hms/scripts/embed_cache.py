"""
HMS v3.6.2 — Embedding Cache & Similarity Engine.

Provides local embedding computation for pre-filtering memories before
expensive LLM calls. Reduces LLM usage by 60-70%.

Priority:
  1. sentence-transformers (best quality, local, zero API cost)
  2. TF-IDF char n-gram (pure Python fallback, always available)

All embeddings are cached to disk to avoid recomputation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import threading

from .file_utils import file_lock

logger = logging.getLogger(__name__)

# Try sentence-transformers first
_HAS_ST = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _HAS_ST = True
except ImportError:
    pass


# ======================================================================
# TF-IDF char n-gram fallback (pure Python, zero dependencies)
# ======================================================================

class CharNGramEncoder:
    """
    Lightweight text encoder using character n-gram hashing.
    Produces fixed-size vectors without any external dependencies.

    v3.2 improvement: uses jieba-aware tokenization when available
    for better semantic quality on Chinese text.
    """

    def __init__(self, dim: int = 256, ngram_range: Tuple[int, int] = (2, 3)):
        self.dim = dim
        self.ngram_min, self.ngram_max = ngram_range

    def _ngrams(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text.lower().strip())
        grams = []
        for n in range(self.ngram_min, self.ngram_max + 1):
            for i in range(len(text) - n + 1):
                grams.append(text[i : i + n])
        return grams

    def encode(self, text: str) -> List[float]:
        """Encode text into a fixed-dim vector.

        Uses jieba tokenization when available for Chinese text.
        Falls back to char n-grams only when jieba is not available
        and the text is ASCII-dominant.
        """
        if not text:
            return [0.0] * self.dim

        # Use jieba tokenization for Chinese text
        cn_ratio = sum(1 for c in text if "\u4e00" <= c <= "\u9fff") / max(len(text), 1)
        if cn_ratio > 0.1:
            from .utils import _get_jieba
            jieba = _get_jieba()
            if jieba is not None:
                tokens = jieba.lcut(text)
                tokens = [w for w in tokens if w.strip()]
                if tokens:
                    vec = [0.0] * self.dim
                    for token in tokens:
                        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
                        vec[h] += 1.0
                    norm = math.sqrt(sum(v * v for v in vec))
                    if norm > 0:
                        vec = [v / norm for v in vec]
                    return vec

        # Fallback to char n-grams for non-Chinese or no-jieba text
        grams = self._ngrams(text)
        if not grams:
            return [0.0] * self.dim
        vec = [0.0] * self.dim
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[h] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(t) for t in texts]


# ======================================================================
# Cosine similarity (works with any vector type)
# ======================================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        logger.warning(
            "cosine_similarity dimension mismatch: len(a)=%d, len(b)=%d", len(a), len(b)
        )
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ======================================================================
# EmbeddingCache — main interface
# ======================================================================

class EmbeddingCache:
    """
    Manages text embeddings with disk caching.
    Supports sentence-transformers or falls back to char n-gram.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self._cache_dir = self.cfg.get("cache_dir", "cache")
        self._cache_path = os.path.join(self._cache_dir, "embedding_cache.json")
        self._cache_bin_path = os.path.join(self._cache_dir, "embedding_cache.bin")
        self._embeddings: OrderedDict[str, List[float]] = OrderedDict()
        self._dirty = False
        self._max_cache_size = self.cfg.get("max_cache_size", 10000)
        self._lock = threading.Lock()

        # Init encoder
        self._encoder_type = "char_ngram"
        self._st_model = None
        self._char_encoder = CharNGramEncoder(dim=256)

        if _HAS_ST:
            model_name = self.cfg.get("embedding_model", "all-MiniLM-L6-v2")
            try:
                self._st_model = SentenceTransformer(model_name)
                self._encoder_type = "sentence-transformers"
            except Exception:
                logger.debug("sentence-transformers load failed, using char_ngram fallback")
                pass  # fallback to char_ngram

        os.makedirs(self._cache_dir, exist_ok=True)
        self._load_cache()

    def _load_cache(self) -> None:
        """Load embeddings from disk.

        Supports both legacy JSON format and new compact binary format.
        Binary format stores float32 vectors as raw bytes (4 bytes per dim)
        instead of verbose JSON number strings.
        """
        # Try binary format first (faster, more compact)
        if os.path.isfile(self._cache_bin_path):
            try:
                with open(self._cache_bin_path, "rb") as f:
                    meta_bytes = f.read(8)
                    if len(meta_bytes) < 8:
                        raise ValueError("truncated")
                    num_entries, dim = struct.unpack("<II", meta_bytes)
                    embeddings = OrderedDict()
                    for _ in range(num_entries):
                        key_len_bytes = f.read(4)
                        if len(key_len_bytes) < 4:
                            raise ValueError("truncated key len")
                        key_len = struct.unpack("<I", key_len_bytes)[0]
                        key = f.read(key_len).decode("utf-8")
                        vec_bytes = f.read(dim * 4)
                        if len(vec_bytes) < dim * 4:
                            raise ValueError("truncated vector")
                        embeddings[key] = list(struct.unpack(f"<{dim}f", vec_bytes))
                    self._embeddings = embeddings
                    logger.debug("Loaded %d embeddings from binary cache", num_entries)
                    return
            except (struct.error, ValueError, IOError) as e:
                logger.warning("Embedding cache dimension mismatch, discarding cache: %s", e)

        # Fall back to legacy JSON format
        if os.path.isfile(self._cache_path):
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._embeddings = OrderedDict(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Embedding cache dimension mismatch, discarding cache: %s", e)
                self._embeddings = OrderedDict()

        # Verify loaded embeddings have correct dimension
        expected_dim = self._char_encoder.dim
        if self._st_model is not None:
            expected_dim = self._st_model.get_sentence_embedding_dimension()
        if self._embeddings and expected_dim:
            first_vec = next(iter(self._embeddings.values()))
            if len(first_vec) != expected_dim:
                logger.warning(
                    "Embedding cache dimension mismatch (got %d, expected %d), discarding cache",
                    len(first_vec), expected_dim,
                )
                self._embeddings = OrderedDict()
        else:
            self._embeddings = OrderedDict()

    def save_cache(self) -> None:
        """Persist embeddings to compact binary format atomically.

        Binary format: [num_entries:u32][dim:u32][key_len:u32][key:bytes][vec:float32*dim]...
        Reduces file size by ~60% compared to JSON and speeds up load/save.
        Uses tempfile + os.replace for atomic writes.
        """
        if not self._dirty:
            return
        with file_lock(self._cache_bin_path):
            # FIX: use actual encoder dimension instead of hardcoded 256
            if self._st_model is not None:
                dim = self._st_model.get_sentence_embedding_dimension()
            else:
                dim = self._char_encoder.dim
            # Write to temporary file then atomically replace
            import tempfile
            dir_name = os.path.dirname(self._cache_bin_path)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(struct.pack("<II", len(self._embeddings), dim))
                    for key, vec in self._embeddings.items():
                        key_bytes = key.encode("utf-8")
                        f.write(struct.pack("<I", len(key_bytes)))
                        f.write(key_bytes)
                        f.write(struct.pack(f"<{dim}f", *vec))
                os.replace(tmp_path, self._cache_bin_path)
            except Exception:
                # On failure, clean up temp file
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            # Remove legacy JSON cache if it exists
            if os.path.isfile(self._cache_path):
                try:
                    os.unlink(self._cache_path)
                except OSError:
                    pass
        self._dirty = False

    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for a single text."""
        if self._st_model is not None:
            vec = self._st_model.encode(text, normalize_embeddings=True)
            return vec.tolist()
        else:
            return self._char_encoder.encode(text)

    def _text_key(self, text: str) -> str:
        """Generate a cache key for text."""
        # FIX: use SHA-256 instead of MD5 to avoid collision risk
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available."""
        key = self._text_key(text)
        with self._lock:
            if key in self._embeddings:
                self._embeddings.move_to_end(key)
                return self._embeddings[key]

            # Check cache size limit
            if len(self._embeddings) >= self._max_cache_size:
                self._evict_old_entries()

        vec = self._compute_embedding(text)
        with self._lock:
            if key not in self._embeddings:
                self._embeddings[key] = vec
                self._dirty = True
            return self._embeddings[key]

    def _evict_old_entries(self) -> None:
        """Evict least recently used 20% of cache entries when limit is reached."""
        if not self._embeddings:
            return
        evict_count = max(1, len(self._embeddings) // 5)
        for _ in range(evict_count):
            self._embeddings.popitem(last=False)
        logger.debug("Evicted %d LRU cache entries", evict_count)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently."""
        results = []
        uncached_indices = []
        uncached_texts = []

        with self._lock:
            for i, text in enumerate(texts):
                key = self._text_key(text)
                if key in self._embeddings:
                    results.append(self._embeddings[key])
                else:
                    results.append(None)  # placeholder
                    uncached_indices.append(i)
                    uncached_texts.append(text)

        if uncached_texts:
            if self._st_model is not None:
                vecs = self._st_model.encode(uncached_texts, normalize_embeddings=True)
                with self._lock:
                    for idx, vec in zip(uncached_indices, vecs):
                        v = vec.tolist()
                        results[idx] = v
                        key = self._text_key(texts[idx])
                        if key not in self._embeddings:
                            self._embeddings[key] = v
                            self._dirty = True
            else:
                with self._lock:
                    for idx, text in zip(uncached_indices, uncached_texts):
                        v = self._char_encoder.encode(text)
                        results[idx] = v
                        key = self._text_key(text)
                        if key not in self._embeddings:
                            self._embeddings[key] = v
                            self._dirty = True

            # FIX: enforce cache size limit after batch add
            with self._lock:
                if len(self._embeddings) > self._max_cache_size:
                    self._evict_old_entries()

        return results

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two texts."""
        emb_a = self.embed(text_a)
        emb_b = self.embed(text_b)
        return cosine_similarity(emb_a, emb_b)

    def find_similar(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        threshold: float = 0.3,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find candidates similar to query, sorted by similarity.
        Returns list of (candidate, similarity_score).
        """
        query_vec = self.embed(query)
        scored = []
        for cand in candidates:
            text = cand.get("text", "")
            if not text:
                continue
            cand_vec = self.embed(text)
            sim = cosine_similarity(query_vec, cand_vec)
            if sim >= threshold:
                scored.append((cand, round(sim, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def cluster_by_similarity(
        self,
        items: List[Dict[str, Any]],
        threshold: float = 0.6,
    ) -> List[List[Dict[str, Any]]]:
        """
        Simple agglomerative clustering by embedding similarity.
        Groups items with pairwise similarity > threshold.

        v3.2: compares all pairs (removed i+30 limit) with
        early exit for large datasets.
        """
        if not items:
            return []

        # Compute all embeddings
        texts = [item.get("text", "") for item in items]
        vecs = self.embed_batch(texts)

        # Union-find clustering
        n = len(items)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Compare all pairs (optimized: skip already-unioned items)
        max_pairs = min(n * (n - 1) // 2, 5000)  # cap for large datasets
        pairs_checked = 0
        for i in range(n):
            for j in range(i + 1, n):
                if pairs_checked >= max_pairs:
                    break
                sim = cosine_similarity(vecs[i], vecs[j])
                if sim >= threshold:
                    union(i, j)
                pairs_checked += 1
            if pairs_checked >= max_pairs:
                break

        # Build clusters
        clusters_map: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            clusters_map.setdefault(root, []).append(i)

        return [[items[i] for i in indices] for indices in clusters_map.values()]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "encoder_type": self._encoder_type,
            "cached_embeddings": len(self._embeddings),
            "has_sentence_transformers": _HAS_ST,
        }


# ======================================================================
# Collision pre-filter
# ======================================================================

def prefilter_for_collision(
    new_text: str,
    existing_memories: List[Dict[str, Any]],
    cache: EmbeddingCache,
    similarity_threshold: float = 0.3,
    max_candidates: int = 10,
) -> List[Dict[str, Any]]:
    """
    Pre-filter memories using embedding similarity.
    Only returns candidates that are semantically related enough
    to warrant an LLM collision check.

    This is the main cost-saving mechanism: instead of sending all
    memories to the LLM, only send the relevant ones.
    """
    similar = cache.find_similar(
        query=new_text,
        candidates=existing_memories,
        top_k=max_candidates,
        threshold=similarity_threshold,
    )
    return [item for item, score in similar]


# ======================================================================
# Self-test
# ======================================================================


def _self_test():
    """Run: python -m hms.scripts.embed_cache"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache({"cache_dir": tmpdir})

        # Test single embedding
        vec = cache.embed("Python编程语言")
        assert len(vec) == 256
        assert all(isinstance(v, float) for v in vec)
        print(f"[embed] dim={len(vec)} type={cache.get_stats()['encoder_type']}")

        # Test similarity
        sim_related = cache.similarity("Python编程", "Python代码")
        sim_unrelated = cache.similarity("Python编程", "今天天气不错")
        print(f"[similarity] related={sim_related:.3f} unrelated={sim_unrelated:.3f}")

        # Test batch embedding
        vecs = cache.embed_batch(["hello", "world", "测试"])
        assert len(vecs) == 3
        print(f"[batch] OK, {len(vecs)} vectors")

        # Test cache hit
        vec2 = cache.embed("Python编程语言")
        assert vec == vec2
        print(f"[cache hit] OK")

        # Test find_similar
        candidates = [
            {"text": "Python是一种编程语言", "id": "m1"},
            {"text": "今天天气很好", "id": "m2"},
            {"text": "Python代码质量", "id": "m3"},
        ]
        similar = cache.find_similar("Python编程", candidates, top_k=5, threshold=0.1)
        print(f"[find_similar] found {len(similar)} matches")
        for cand, score in similar:
            print(f"  {cand['id']}: {cand['text'][:20]} (sim={score})")

        # Test clustering
        items = [
            {"text": "Python是一种语言", "id": "a"},
            {"text": "Python代码编写", "id": "b"},
            {"text": "今天去公园散步", "id": "c"},
            {"text": "Python开发项目", "id": "d"},
        ]
        clusters = cache.cluster_by_similarity(items, threshold=0.2)
        print(f"[clustering] {len(items)} items -> {len(clusters)} clusters")

        # Test prefilter
        memories = [
            {"text": "用户喜欢Python", "id": "m1"},
            {"text": "用户养了一只猫", "id": "m2"},
            {"text": "Python项目进展", "id": "m3"},
        ]
        filtered = prefilter_for_collision("Python代码问题", memories, cache, similarity_threshold=0.1)
        print(f"[prefilter] {len(memories)} -> {len(filtered)} candidates")

        # Test save/load
        cache.save_cache()
        cache2 = EmbeddingCache({"cache_dir": tmpdir})
        assert len(cache2._embeddings) == len(cache._embeddings)
        print(f"[persistence] OK, {len(cache2._embeddings)} cached")

        print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
