from pathlib import Path

from agent_service.rag.chunking import chunk_markdown, load_corpus

POLICY_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"


def test_chunks_are_section_level_with_citations():
    chunks = chunk_markdown(POLICY_DIR / "POL-100_three_way_matching.md")
    assert len(chunks) >= 4
    tol = next(c for c in chunks if "Matching tolerances" in c.section)
    assert tol.doc_id == "POL-100" and "2%" in tol.text
    assert tol.citation == "[POL-100 2. Matching tolerances]"


def test_agreements_are_typed():
    assert {c.doc_type for c in load_corpus(POLICY_DIR)} == {"policy", "agreement"}


def test_long_sections_split_with_overlap(tmp_path):
    f = tmp_path / "POL-999_long.md"
    sentences = " ".join(f"Sentence number {i} is here." for i in range(60))
    f.write_text(f"# Long doc\n\n## 1. Big section\n{sentences}\n")
    chunks = chunk_markdown(f, max_chars=200)
    assert len(chunks) > 2
    assert all(len(c.text) < 400 for c in chunks)


def _top_ids(retriever, query, k=3, **kw):
    return [h.chunk.doc_id + " " + h.chunk.section for h in retriever.search(query, k=k, **kw)]


def test_price_tolerance_question_finds_matching_policy(retriever):
    assert any("POL-100" in c and "tolerances" in c for c in _top_ids(retriever, "what is the unit price tolerance"))


def test_approval_question_finds_authority_section(retriever):
    assert any("POL-300" in c and "Approval authority" in c for c in _top_ids(retriever, "who approves a discrepancy of CAD 6000"))


def test_duplicate_question_finds_duplicate_policy(retriever):
    assert any(c.startswith("POL-400") for c in _top_ids(retriever, "what do we do with a duplicate invoice"))


def test_late_penalty_question_finds_delay_clause(retriever):
    assert any("POL-200" in c and "penalties" in c for c in _top_ids(retriever, "late delivery penalty per day"))


def test_doc_type_filter(retriever):
    hits = retriever.search("late delivery", k=4, doc_type="agreement")
    assert hits and all(h.chunk.doc_type == "agreement" for h in hits)


def test_hybrid_fuses_both_rankers(retriever):
    hits = retriever.search("unit price tolerance", k=4)
    assert any(h.vector_rank and h.bm25_rank for h in hits)
    assert all(h.score > 0 for h in hits)
