"""Hybrid retrieval: ChromaDB vector search + BM25 keyword search, merged with
Reciprocal Rank Fusion (RRF). Supports metadata filtering and returns citations."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from .chunking import Chunk, load_corpus
from .embeddings import Embedder

_TOKEN = re.compile(r"[a-z0-9]+")
COLLECTION = "procurement_docs"


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Hit:
    chunk: Chunk
    score: float
    vector_rank: int | None
    bm25_rank: int | None

    @property
    def citation(self) -> str:
        return self.chunk.citation


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, persist_dir: str | None = None) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self._by_id = {c.chunk_id: c for c in chunks}
        self._bm25 = BM25Okapi([_tok(c.text) for c in chunks])

        self._client = (
            chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        )
        # Rebuild the collection so the index always matches the corpus on disk.
        try:
            self._client.delete_collection(COLLECTION)
        except Exception:
            pass
        self._col = self._client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embedder.embed([c.text for c in chunks]),
            metadatas=[
                {"doc_id": c.doc_id, "doc_type": c.doc_type, "section": c.section, "source": c.source}
                for c in chunks
            ],
        )

    @classmethod
    def from_directory(cls, directory: Path, embedder: Embedder, persist_dir: str | None = None,
                       max_chars: int = 600) -> "HybridRetriever":
        return cls(load_corpus(directory, max_chars=max_chars), embedder, persist_dir)

    def search(self, query: str, k: int = 4, doc_type: str | None = None, rrf_k: int = 60,
               candidates: int = 10) -> list[Hit]:
        allowed = {c.chunk_id for c in self.chunks if doc_type in (None, c.doc_type)}

        # keyword ranking
        scores = self._bm25.get_scores(_tok(query))
        bm25_order = sorted(
            (i for i, c in enumerate(self.chunks) if c.chunk_id in allowed and scores[i] > 0),
            key=lambda i: scores[i], reverse=True,
        )[:candidates]
        bm25_rank = {self.chunks[i].chunk_id: r for r, i in enumerate(bm25_order, start=1)}

        # vector ranking
        where = {"doc_type": doc_type} if doc_type else None
        res = self._col.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=min(candidates, len(allowed)) or 1,
            where=where,
        )
        vec_rank = {cid: r for r, cid in enumerate(res["ids"][0], start=1)}

        # reciprocal rank fusion
        fused: dict[str, float] = {}
        for ranks in (bm25_rank, vec_rank):
            for cid, r in ranks.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + r)

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(self._by_id[cid], s, vec_rank.get(cid), bm25_rank.get(cid)) for cid, s in top]
