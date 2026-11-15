"""Embedding backends.

* SentenceTransformerEmbedder - real semantic embeddings (default when available).
* HashingEmbedder - deterministic, offline, dependency-free feature hashing. It is LEXICAL,
  not semantic. It exists so tests and offline demos run without downloading a model.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Protocol

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    name = "hashing-384"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        toks = _TOKEN.findall(text.lower())
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            for f in self._features(t):
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                sign = 1.0 if (h >> 1) & 1 else -1.0
                vec[h % self.dim] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # heavy, optional import

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def get_embedder() -> Embedder:
    """EMBEDDING_BACKEND = auto (default) | sentence-transformers | hashing."""
    backend = os.getenv("EMBEDDING_BACKEND", "auto").lower()
    if backend == "hashing":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(
            os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
    except Exception as exc:  # ImportError, offline model download, etc.
        if backend == "sentence-transformers":
            raise
        log.warning("Falling back to hashing embeddings (lexical only): %s", exc)
        return HashingEmbedder()
