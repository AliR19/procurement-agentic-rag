"""Heading-aware chunking for markdown documents.

Each `## section` becomes one chunk (prefixed with the document title so the chunk still
makes sense on its own). Sections longer than `max_chars` are split on sentence boundaries
with a one-sentence overlap so facts that straddle a split are not lost.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    doc_type: str  # "policy" or "agreement"
    title: str
    section: str
    source: str
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"[{self.doc_id} {self.section}]" if self.section else f"[{self.doc_id}]"


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = _SENTENCE.split(text)
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for s in sentences:
        if current and size + len(s) > max_chars:
            parts.append(" ".join(current))
            current = [current[-1], s]  # one-sentence overlap
            size = len(current[0]) + len(s)
        else:
            current.append(s)
            size += len(s) + 1
    if current:
        parts.append(" ".join(current))
    return parts


def doc_id_from_path(path: Path) -> str:
    return path.stem.split("_")[0]  # POL-100_three_way... -> POL-100


def chunk_markdown(path: Path, max_chars: int = 600) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    doc_id = doc_id_from_path(path)
    doc_type = "agreement" if doc_id.startswith("AGR") else "policy"

    title_match = re.search(r"^#\s+(.+)$", raw, flags=re.M)
    title = title_match.group(1).strip() if title_match else doc_id

    chunks: list[Chunk] = []
    sections = re.split(r"^##\s+", raw, flags=re.M)[1:]
    for sec in sections:
        heading, _, body = sec.partition("\n")
        heading = heading.strip()
        body = body.strip()
        if not body:
            continue
        for i, piece in enumerate(_split_long(body, max_chars)):
            text = f"{title} - {heading}\n{piece}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{heading[:2].strip('. ')}:{i}",
                    text=text,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    title=title,
                    section=heading,
                    source=path.name,
                )
            )
    return chunks


def load_corpus(directory: Path, max_chars: int = 600) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_markdown(path, max_chars=max_chars))
    return chunks
