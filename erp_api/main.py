"""Mock ERP / procurement API.

A small stand-in for an enterprise business system (think purchase orders, invoices and
suppliers behind a REST API). The agent talks to it ONLY through this HTTP interface, the
same way it would talk to a real ERP. All data is simulated (see scripts/generate_data.py).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

SEED_FILE = Path(__file__).resolve().parent / "seed.json"
API_KEY = os.getenv("ERP_API_KEY", "dev-key")

app = FastAPI(title="Mock ERP API (simulated data)", version="1.0.0")

_DATA: dict[str, Any] = json.loads(SEED_FILE.read_text())
_EXCEPTION_REPORTS: list[dict[str, Any]] = []


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")


class ExceptionReportIn(BaseModel):
    po_number: str
    invoice_number: str | None = None
    summary: str = Field(min_length=5)
    discrepancies: list[dict[str, Any]] = Field(default_factory=list)
    requested_action: str = "Review and approve resolution"
    approved_by: str = "human reviewer (demo: identity not recorded)"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/suppliers/{supplier_id}", dependencies=[Depends(require_key)])
def get_supplier(supplier_id: str) -> dict[str, Any]:
    for s in _DATA["suppliers"]:
        if s["supplier_id"] == supplier_id:
            return s
    raise HTTPException(404, f"Supplier {supplier_id} not found")


@app.get("/purchase-orders/{po_number}", dependencies=[Depends(require_key)])
def get_purchase_order(po_number: str) -> dict[str, Any]:
    for po in _DATA["purchase_orders"]:
        if po["po_number"] == po_number:
            return po
    raise HTTPException(404, f"Purchase order {po_number} not found")


@app.get("/invoices", dependencies=[Depends(require_key)])
def list_invoices(po_number: str) -> list[dict[str, Any]]:
    return [i for i in _DATA["invoices"] if i["po_number"] == po_number]


@app.post("/exception-reports", status_code=201, dependencies=[Depends(require_key)])
def create_exception_report(body: ExceptionReportIn) -> dict[str, Any]:
    report = {
        "report_id": f"EXR-{len(_EXCEPTION_REPORTS) + 1:04d}",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "Submitted",
        **body.model_dump(),
    }
    _EXCEPTION_REPORTS.append(report)
    return report


@app.get("/exception-reports", dependencies=[Depends(require_key)])
def list_exception_reports() -> list[dict[str, Any]]:
    return _EXCEPTION_REPORTS
