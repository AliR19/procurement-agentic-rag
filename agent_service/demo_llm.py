"""DEMO MODE: a deterministic, rule-based stand-in for an LLM.

This is NOT a language model. It reads the conversation, picks tools with simple rules
(regex for PO numbers, keywords) and writes a templated answer from the tool results.
It exists so the whole system (tools, RAG, graph, approval gate, API, UI) can run and be
tested with no model, no API key and no network. For real language understanding, set
LLM_PROVIDER=ollama or LLM_PROVIDER=openai.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_PO = re.compile(r"\b(45\d{4})\b")
_SUPPLIER = re.compile(r"\b(S\d{4})\b", re.I)
_FILE_REPORT = re.compile(
    r"(create|file|submit|raise|open|draft).{0,30}(exception|report|approval)|^\s*(yes|yep|yeah|please do|go ahead|do it)\b",
    re.I,
)


def _call(name: str, args: dict[str, Any], n: int) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"demo-{n}", "type": "tool_call"}


def _tool_results(messages: list[BaseMessage]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m in messages:
        if isinstance(m, ToolMessage):
            try:
                out[m.name or ""] = json.loads(m.content)
            except (TypeError, ValueError):
                out[m.name or ""] = m.content
    return out


def _snip(text: str, limit: int = 260) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rsplit(" ", 1)[0] + "..."


def _money(x: float) -> str:
    return f"CAD {x:,.2f}"


class DemoChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "demo-rule-based"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "DemoChatModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._respond(messages))])

    # ------------------------------------------------------------------ logic
    def _respond(self, messages: list[BaseMessage]) -> AIMessage:
        last_h = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        text = str(messages[last_h].content)
        turn = _tool_results(messages[last_h + 1:])
        history = _tool_results(messages[:last_h])

        # 1) user asks to file the report
        if _FILE_REPORT.search(text) and "compare_po_and_invoice" in history:
            if "create_exception_report" in turn:
                return AIMessage(content=self._report_reply(turn["create_exception_report"]))
            cmp_ = history["compare_po_and_invoice"]
            kinds = ", ".join(sorted({d["type"].replace("_", " ") for d in cmp_.get("discrepancies", [])})) or "mismatch"
            return AIMessage(content="Filing the exception report now. It needs human approval first.", tool_calls=[
                _call("create_exception_report", {
                    "po_number": cmp_["po_number"],
                    "invoice_number": (cmp_.get("invoice_numbers") or [""])[0],
                    "summary": f"Three-way match failed for PO {cmp_['po_number']}: {kinds}. "
                               f"Financial impact {_money(cmp_.get('total_financial_impact', 0))}.",
                    "discrepancies": cmp_.get("discrepancies", []),
                    "requested_action": f"Review and approve resolution ({cmp_.get('approval_authority', 'approver')})",
                }, 1)])

        # 2) PO investigation
        po = _PO.search(text)
        if po:
            n = po.group(1)
            if "compare_po_and_invoice" not in turn:
                return AIMessage(content=f"Checking PO {n}.", tool_calls=[
                    _call("get_purchase_order", {"po_number": n}, 1),
                    _call("get_invoice_details", {"po_number": n}, 2),
                    _call("compare_po_and_invoice", {"po_number": n}, 3)])
            if "search_procurement_policy" not in turn:
                cmp_ = turn["compare_po_and_invoice"]
                kinds = {d["type"] for d in cmp_.get("discrepancies", [])}
                q = []
                if "price_variance" in kinds: q.append("unit price tolerance price increase")
                if "quantity_variance" in kinds: q.append("quantity tolerance short delivery")
                if "late_delivery" in kinds: q.append("late delivery grace period penalty")
                if "duplicate_invoice" in kinds: q.append("duplicate invoice")
                if kinds: q.append("exception report approval authority")
                calls = [_call("search_procurement_policy", {"query": " ".join(q) or "three-way matching tolerances"}, 4)]
                po_data = turn.get("get_purchase_order", {})
                if isinstance(po_data, dict) and po_data.get("supplier_id"):
                    calls.append(_call("get_supplier_information", {"supplier_id": po_data["supplier_id"]}, 5))
                return AIMessage(content="Looking up the relevant policy.", tool_calls=calls)
            return AIMessage(content=self._po_reply(turn))

        # 3) supplier lookup
        sup = _SUPPLIER.search(text)
        if sup:
            if "get_supplier_information" not in turn:
                return AIMessage(content="", tool_calls=[_call("get_supplier_information", {"supplier_id": sup.group(1).upper()}, 1)])
            return AIMessage(content="Supplier record:\n```json\n" + json.dumps(turn["get_supplier_information"], indent=2) + "\n```")

        # 4) general policy question
        if "search_procurement_policy" not in turn:
            return AIMessage(content="", tool_calls=[_call("search_procurement_policy", {"query": text}, 1)])
        return AIMessage(content=self._policy_reply(turn["search_procurement_policy"]))

    # ------------------------------------------------------------ templates
    @staticmethod
    def _policy_reply(hits: list[dict[str, str]]) -> str:
        if not hits:
            return "I could not find a matching policy."
        lines = ["Most relevant policy passages:"]
        for h in hits[:3]:
            lines.append(f"- {h['citation']} {_snip(h['text'])}")
        return "\n".join(lines)

    @staticmethod
    def _report_reply(res: Any) -> str:
        if isinstance(res, str) and res.startswith("REJECTED"):
            return "The exception report was not filed because the reviewer rejected it. Nothing was written to the ERP."
        return (f"Exception report **{res.get('report_id')}** was filed for PO {res.get('po_number')} "
                f"(status: {res.get('status')}).")

    @staticmethod
    def _po_reply(turn: dict[str, Any]) -> str:
        po = turn.get("get_purchase_order", {})
        cmp_ = turn["compare_po_and_invoice"]
        sup = turn.get("get_supplier_information", {})
        hits = turn.get("search_procurement_policy", [])
        if "error" in po:
            return f"I could not find that purchase order: {po['error']}."

        head = f"**PO {po['po_number']}** - {sup.get('name', po['supplier_id'])}, status {po['status']}, total {_money(po['total'])}."
        if cmp_["status"] == "not_invoiced":
            return head + "\n\nNo invoice has been received yet, so there is nothing to match."
        if cmp_["status"] == "match":
            return head + f"\n\nInvoice {cmp_['invoice_numbers'][0]} matches the PO within policy tolerances. No action needed."

        lines = [head, f"\nInvoice **{', '.join(cmp_['invoice_numbers'])}** has discrepancies:"]
        for d in cmp_["discrepancies"]:
            t = d["type"]
            if t == "price_variance":
                lines.append(f"- Line {d['line']} price: PO {_money(d['po_unit_price'])} vs invoice {_money(d['invoice_unit_price'])} "
                             f"({d['variance_pct']:+.1f}%), impact {_money(d['financial_impact'])}.")
            elif t == "quantity_variance":
                lines.append(f"- Line {d['line']} quantity: ordered {d['ordered_quantity']}, invoiced {d['invoiced_quantity']} "
                             f"({d['variance_pct']:+.1f}%, {d['direction']}).")
            elif t == "late_delivery":
                lines.append(f"- Delivered {d['business_days_late']} business days late (expected {d['expected_delivery']}, received {d['actual_delivery']}).")
            elif t == "duplicate_invoice":
                lines.append(f"- Possible duplicate invoices: {', '.join(d['invoice_numbers'])}.")
            else:
                lines.append(f"- {t}: {d.get('detail', '')}")
        if "total_difference" in cmp_:
            lines.append(f"\nPO total {_money(cmp_['po_total'])} vs invoice total {_money(cmp_['invoice_total'])} "
                         f"(difference {_money(cmp_['total_difference'])}).")
        lines.append(f"Estimated financial impact: **{_money(cmp_['total_financial_impact'])}**. "
                     f"Approval authority: **{cmp_.get('approval_authority', 'n/a')}**.")
        if hits:
            lines.append("\n**Relevant policy**")
            for h in hits[:3]:
                lines.append(f"- {h['citation']} {_snip(h['text'])}")
        lines.append("\nWould you like me to file an exception report? It will need human approval.")
        return "\n".join(lines)
