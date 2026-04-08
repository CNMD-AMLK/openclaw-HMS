"""
HMS v5 — Embedding Cache (adapted from hms-v4 scripts/embed_cache.py).
Simplified to focus on Ollama API + char-ngram fallback.
"""

from __future__ import annotations
import hashlib
import json
import math
import os
import struct
import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from hms.utils.file_utils import file_lock

logger = logging.getLogger("hms.embed")

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class CharNGramEncoder:
    def __init__(self, dim: int = 256, ngram_range=(2, 3)):
        self.dim = dim
        self.ngram_min, self.ngram_max = ngram_range

    def encode(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dim
        text = text.lower().strip()
        import hashlib as _hm
        vec = [0.0] * self.dim
        for n in range(self.ngram_min, self.ngram_max + 1):
            for i in range(len(text) - n + 1):
                g = text[i:i + n]
                h = int(_hm.md5(g.encode("utf-8")).hexdigest(), 16) % self.dim
                vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class EmbeddingCache:
    """
    Manages text embeddings with Ollama API + disk caching.
    Primary: Ollama /v1/embeddings (qwen3-embedding:0.6b)
    Fallback: char n-gram
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = config or {}
        self._cache_dir = self.cfg.get("cache_dir", "cache")
        self._cache_bin_path = os.path.join(self._cache_dir, "embeddings.bin")
        self._embeddings: OrderedDict[str, List[float]] = OrderedDict()
        self._dirty = False
        self._max_cache_size = self.cfg.get("max_cache_size", 10000)
        self._lock = threading.Lock()

        self._encoder_type = "char_ngram"
        self._char_encoder = CharNGramEncoder(dim=256)
        self._ollama_base_url = None
        self._ollama_model = None
        self._ollama_dim = 1024

        if _HAS_REQUESTS:
            base_url = self.cfg.get("embedding_ollama_base_url", "").rstrip("/")
            model = self.cfg.get("embedding_ollama_model", "qwen3-embedding:0.6b")
            if base_url and model:
                self._ollama_base_url = base_url
                self._ollama_model = model
                self._ollama_dim = self.cfg.get("embedding_ollama_dim", 1024)
                self._encoder_type = "ollama"

        os.makedirs(self._cache_dir, exist_ok=True)
        self._load_cache()

    def _load_cache(self) -> None:
        if os.path.isfile(self._cache_bin_path):
            try:
                with open(self._cache_bin_path, "rb") as f:
                    meta = struct.unpack("<II", f.read(8))
                    num_entries, dim = meta
                    embeddings = OrderedDict()
                    for _ in range(num_entries):
                        key_len = struct.unpack("<I", f.read(4))[0]
                        key = f.read(key_len).decode("utf-8")
                        vec = list(struct.unpack(f"<{dim}f", f.read(dim * 4)))
                        embeddings[key] = vec
                    self._embeddings = embeddings
                    return
            except Exception as e:
                logger.warning("Failed to load embedding cache: %s", e)

        if os.path.isfile(self._cache_bin_path.replace(".bin", ".json")):
            try:
                with open(self._cache_bin_path.replace(".bin", ".json"), encoding="utf-8") as f:
                    self._embeddings = OrderedDict(json.load(f))
            except Exception:
                self._embeddings = OrderedDict()

    def save_cache(self) -> None:
        if not self._dirty:
            return
        with file_lock(self._cache_bin_path):
            dim = self._ollama_dim if self._encoder_type == "ollama" else 256
            import tempfile
            dir_name = os.path.dirname(self._cache_bin_path) or "."
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
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        self._dirty = False

    def _text_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> List[float]:
        key = self._text_key(text)
        with self._lock:
            if key in self._embeddings:
                self._embeddings.move_to_end(key)
                return self._embeddings[key]

        if len(self._embeddings) >= self._max_cache_size:
            self._evict_old_entries()

        vec = self._compute_embedding(text)
        with self._lock:
            if key not in self._embeddings:
                self._embeddings[key] = vec
                self._dirty = True
        return vec

    def _compute_embedding(self, text: str) -> List[float]:
        if self._ollama_base_url and _HAS_REQUESTS:
            try:
                resp = requests.post(
                    f"{self._ollama_base_url}/v1/embeddings",
                    json={"model": self._ollama_model, "input": text},
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["data"][0]["embedding"]
            except Exception as e:
                logger.warning("Ollama embed failed: %s, using char-ngram", e)
                self._ollama_base_url = None
                self._encoder_type = "char_ngram"
        return self._char_encoder.encode(text)

    def _evict_old_entries(self) -> None:
        evict_count = max(1, len(self._embeddings) // 5)
        for _ in range(evict_count):
            self._embeddings.popitem(last=False)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            results.append(self.embed(text))
        return results

    def similarity(self, text_a: str, text_b: str) -> float:
        emb_a = self.embed(text_a)
        emb_b = self.embed(text_b)
        return cosine_similarity(emb_a, emb_b)

    def find_similar(self, query: str, candidates: List[Dict[str, Any]],
                     top_k: int = 10, threshold: float = 0.3) -> List[tuple]:
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

    def get_stats(self) -> Dict[str, Any]:
        return {
            "encoder_type": self._encoder_type,
            "cached_embeddings": len(self._embeddings),
        }
