"""Deterministic guardrails around the LLM.

A small model can skip a tool, quote a citation it never retrieved, or misreport what just happened.
These checks do not rely on the model behaving:

* missing_policy_search : after a failed PO/invoice match, the policy search must happen. If the
                          model did not ask for it, the graph runs it.
* verify_citations      : a citation such as [POL-100 2. Matching tolerances] may only appear in an
                          answer if search_procurement_policy actually returned it in this conversation.
* Sources line          : if the policy was searched for this answer but the model cited nothing, the
                          retrieved citations are listed, so an answer is never left without its sources.
* write_outcome         : after the write tool runs (or is rejected) the answer must state the real
                          outcome. If the model's message does not, it is replaced by the true one.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

POLICY_TOOL = "search_procurement_policy"
COMPARE_TOOL = "compare_po_and_invoice"
WRITE_TOOL = "create_exception_report"

_CITATION = re.compile(r"\[(?:POL|AGR)-\d+[^\]\n]*\]")
_DOC_ONLY = re.compile(r"\[((?:POL|AGR)-\d+)\]")


def _norm(text: str) -> str:
    return " ".join(text.split())


def _json(message: ToolMessage) -> Any:
    try:
        return json.loads(message.content)
    except (TypeError, ValueError):
        return None


def _current_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    return messages[start + 1:]


def retrieved_citations(messages: list[BaseMessage]) -> set[str]:
    """Every citation string returned by the policy search anywhere in the conversation."""
    found: set[str] = set()
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == POLICY_TOOL:
            data = _json(m)
            if isinstance(data, list):
                found.update(_norm(h["citation"]) for h in data if isinstance(h, dict) and h.get("citation"))
    return found


def turn_citations(messages: list[BaseMessage]) -> list[str]:
    """Citations the policy search returned in the current turn, in retrieval order."""
    seen: list[str] = []
    for m in _current_turn(messages):
        if isinstance(m, ToolMessage) and m.name == POLICY_TOOL:
            data = _json(m)
            for h in data if isinstance(data, list) else []:
                cite = _norm(h.get("citation", "")) if isinstance(h, dict) else ""
                if cite and cite not in seen:
                    seen.append(cite)
    return seen


def policy_query_for(comparison: dict[str, Any]) -> str:
    kinds = {d.get("type") for d in comparison.get("discrepancies", [])}
    parts = []
    if "price_variance" in kinds:
        parts.append("unit price tolerance price increase")
    if "quantity_variance" in kinds:
        parts.append("quantity tolerance short delivery")
    if "late_delivery" in kinds:
        parts.append("late delivery grace period penalty")
    if "duplicate_invoice" in kinds:
        parts.append("duplicate invoice")
    parts.append("exception report approval authority")
    return " ".join(parts)


def missing_policy_search(messages: list[BaseMessage]) -> str | None:
    """A query to run if this turn compared a PO with discrepancies but never searched the policy."""
    turn = _current_turn(messages)
    if any(isinstance(m, ToolMessage) and m.name == POLICY_TOOL for m in turn):
        return None
    for m in turn:
        if isinstance(m, ToolMessage) and m.name == COMPARE_TOOL:
            data = _json(m)
            if isinstance(data, dict) and data.get("discrepancies"):
                return policy_query_for(data)
    return None


def verify_citations(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    """Remove citations that the policy search never returned. Returns (text, removed)."""
    removed: list[str] = []

    def check(match: re.Match[str]) -> str:
        cite = _norm(match.group(0))
        if cite in allowed:
            return match.group(0)
        doc = _DOC_ONLY.fullmatch(cite)
        if doc and any(a.startswith(f"[{doc.group(1)} ") for a in allowed):
            return match.group(0)  # "[POL-100]" is fine when a section of POL-100 was retrieved
        removed.append(cite)
        return ""

    cleaned = _CITATION.sub(check, text)
    if removed:
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)
        cleaned = cleaned.rstrip() + (
            f"\n\n(Note: {len(removed)} policy citation(s) in the draft answer did not match any retrieved "
            "policy text and were removed.)")
    return cleaned, removed


def write_outcome(messages: list[BaseMessage]) -> tuple[str, str, str | None] | None:
    """(kind, true message, marker) if the last message is the result of the write tool."""
    last = messages[-1] if messages else None
    if not (isinstance(last, ToolMessage) and last.name == WRITE_TOOL):
        return None
    if isinstance(last.content, str) and last.content.startswith("REJECTED"):
        return ("rejected",
                "The exception report was not filed because the reviewer rejected it. Nothing was written to the ERP.",
                None)
    data = _json(last)
    if isinstance(data, dict) and data.get("report_id"):
        return ("filed",
                f"Exception report {data['report_id']} was filed for PO {data.get('po_number')} "
                f"(status: {data.get('status')}).",
                str(data["report_id"]))
    return None  # a tool error: let the model explain it


def finalize_reply(reply: AIMessage, history: list[BaseMessage]) -> AIMessage:
    """Apply the output guardrails to a final (no tool call) answer."""
    if reply.tool_calls or not isinstance(reply.content, str):
        return reply
    text = reply.content
    info: dict[str, Any] = {"citations_removed": [], "outcome_replaced": False, "sources_appended": False}

    outcome = write_outcome(history)
    if outcome:
        kind, truth, marker = outcome
        low = text.lower()
        stated = (marker in text) if kind == "filed" else any(
            w in low for w in ("reject", "declin", "not filed", "nothing was written"))
        if not stated:
            text, info["outcome_replaced"] = truth, True

    text, info["citations_removed"] = verify_citations(text, retrieved_citations(history))

    retrieved_now = turn_citations(history)
    if retrieved_now and not _CITATION.search(text):  # the policy was searched but the answer cites nothing
        text = text.rstrip() + "\n\nPolicy passages retrieved for this answer: " + "; ".join(retrieved_now[:3])
        info["sources_appended"] = True
    return reply.model_copy(update={"content": text,
                                    "additional_kwargs": {**reply.additional_kwargs, "guardrails": info}})
