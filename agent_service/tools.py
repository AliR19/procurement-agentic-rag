"""Agent tools.

Read tools call the ERP over HTTP (never touching its data directly). `search_procurement_policy`
uses the hybrid RAG retriever. `create_exception_report` WRITES to the ERP, so the agent graph
requires explicit human approval before it runs (see agent.py).
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
from langchain_core.tools import BaseTool, tool

from .matching import compare
from .rag.hybrid import HybridRetriever

SENSITIVE_TOOLS = {"create_exception_report"}


class ErpClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 client: httpx.Client | None = None) -> None:
        headers = {"x-api-key": api_key or os.getenv("ERP_API_KEY", "dev-key")}
        self._client = client or httpx.Client(
            base_url=base_url or os.getenv("ERP_BASE_URL", "http://localhost:8001"), timeout=10.0
        )
        self._headers = headers

    def get(self, path: str, **params: Any) -> Any:
        r = self._client.get(path, params=params or None, headers=self._headers)
        if r.status_code == 404:
            return {"error": r.json().get("detail", "not found")}
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: dict[str, Any]) -> Any:
        r = self._client.post(path, json=body, headers=self._headers)
        r.raise_for_status()
        return r.json()


def build_tools(erp: ErpClient, retriever: HybridRetriever) -> list[BaseTool]:
    @tool
    def get_purchase_order(po_number: str) -> str:
        """Fetch a purchase order (header, supplier id, delivery dates, lines, total) by PO number."""
        return json.dumps(erp.get(f"/purchase-orders/{po_number}"))

    @tool
    def get_invoice_details(po_number: str) -> str:
        """Fetch all supplier invoices recorded against a PO number."""
        invoices = erp.get("/invoices", po_number=po_number)
        return json.dumps(invoices if invoices else {"note": f"No invoice found for PO {po_number}"})

    @tool
    def get_supplier_information(supplier_id: str) -> str:
        """Fetch supplier details (name, payment terms, rating, contract id) by supplier id such as S1008."""
        return json.dumps(erp.get(f"/suppliers/{supplier_id}"))

    @tool
    def compare_po_and_invoice(po_number: str) -> str:
        """Run the three-way-match checks for a PO: price, quantity, total, late delivery and
        duplicate invoices. Returns discrepancies with financial impact. Use this instead of
        comparing numbers yourself."""
        po = erp.get(f"/purchase-orders/{po_number}")
        if "error" in po:
            return json.dumps(po)
        invoices = erp.get("/invoices", po_number=po_number)
        return json.dumps(compare(po, invoices))

    @tool
    def search_procurement_policy(query: str, doc_type: str = "") -> str:
        """Search procurement policies and supplier agreements (hybrid keyword + vector search).
        Returns passages with citations like [POL-100 2. Matching tolerances]. Optional doc_type:
        'policy' or 'agreement'. Always cite these citations when explaining a rule."""
        hits = retriever.search(query, k=4, doc_type=doc_type or None)
        return json.dumps([
            {"citation": h.citation, "text": h.chunk.text.split("\n", 1)[-1], "source": h.chunk.source}
            for h in hits
        ])

    @tool
    def create_exception_report(po_number: str, invoice_number: str, summary: str,
                                discrepancies: list[dict[str, Any]], requested_action: str = "") -> str:
        """File an exception report in the ERP for a PO/invoice discrepancy. This WRITES a record and
        requires human approval; only call it after the user has asked for the report."""
        body = {"po_number": po_number, "invoice_number": invoice_number, "summary": summary,
                "discrepancies": discrepancies}
        if requested_action:
            body["requested_action"] = requested_action
        return json.dumps(erp.post("/exception-reports", body))

    return [get_purchase_order, get_invoice_details, get_supplier_information,
            compare_po_and_invoice, search_procurement_policy, create_exception_report]
