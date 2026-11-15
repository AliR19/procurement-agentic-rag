"""Deterministic three-way-match checks between a purchase order and its invoice(s).

The arithmetic is done in plain Python on purpose: an LLM should explain and route, not do
money math. The thresholds below mirror the fictional policies in data/policies
(POL-100 matching tolerances, POL-200 grace period, POL-300 approval authority).
In a real system they would come from governed configuration, not constants.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

PRICE_TOLERANCE = 0.02       # POL-100 s2: unit price within 2%
QUANTITY_TOLERANCE = 0.05    # POL-100 s2: quantity within 5%
TOTAL_TOLERANCE_CAD = 50.0   # POL-100 s2: total within CAD 50
GRACE_BUSINESS_DAYS = 3      # POL-200 s2


def business_days_late(expected: str, actual: str | None) -> int:
    if not actual:
        return 0
    start, end = date.fromisoformat(expected), date.fromisoformat(actual)
    days, d = 0, start
    while d < end:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days += 1
    return days


def approval_authority(impact_cad: float) -> str:
    """POL-300 s3."""
    if impact_cad >= 5000:
        return "Procurement director"
    if impact_cad >= 500:
        return "Procurement manager"
    return "Accounts payable supervisor"


def _pct(new: float, old: float) -> float:
    return round((new - old) / old * 100, 2) if old else 0.0


def compare(po: dict[str, Any], invoices: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "po_number": po["po_number"],
        "invoice_numbers": [i["invoice_number"] for i in invoices],
        "discrepancies": [],
        "total_financial_impact": 0.0,
    }

    if not invoices:
        result["status"] = "not_invoiced"
        return result

    inv = invoices[0]
    po_lines = {l["line"]: l for l in po["lines"]}
    discrepancies: list[dict[str, Any]] = []

    for line in inv["lines"]:
        pl = po_lines.get(line["line"])
        if not pl:
            discrepancies.append({"type": "unexpected_line", "line": line["line"],
                                  "detail": "Invoice line has no matching PO line", "financial_impact": 0.0})
            continue

        price_pct = _pct(line["unit_price"], pl["unit_price"])
        if abs(price_pct) > PRICE_TOLERANCE * 100:
            impact = round((line["unit_price"] - pl["unit_price"]) * line["quantity"], 2)
            discrepancies.append({
                "type": "price_variance", "line": line["line"], "sku": pl["sku"],
                "po_unit_price": pl["unit_price"], "invoice_unit_price": line["unit_price"],
                "variance_pct": price_pct, "financial_impact": impact,
            })

        qty_pct = _pct(line["quantity"], pl["quantity"])
        if abs(qty_pct) > QUANTITY_TOLERANCE * 100:
            over = line["quantity"] > pl["quantity"]
            discrepancies.append({
                "type": "quantity_variance", "line": line["line"], "sku": pl["sku"],
                "ordered_quantity": pl["quantity"], "invoiced_quantity": line["quantity"],
                "variance_pct": qty_pct, "direction": "over" if over else "short",
                "financial_impact": round((line["quantity"] - pl["quantity"]) * line["unit_price"], 2) if over else 0.0,
            })

    late = business_days_late(po["expected_delivery"], po.get("actual_delivery"))
    if late > GRACE_BUSINESS_DAYS:
        discrepancies.append({
            "type": "late_delivery", "expected_delivery": po["expected_delivery"],
            "actual_delivery": po["actual_delivery"], "business_days_late": late,
            "financial_impact": 0.0,
        })

    if len(invoices) > 1:
        discrepancies.append({
            "type": "duplicate_invoice",
            "invoice_numbers": [i["invoice_number"] for i in invoices],
            "detail": "More than one invoice exists for this PO",
            "financial_impact": invoices[1]["total"],
        })

    total_diff = round(inv["total"] - po["total"], 2)
    result.update({
        "po_total": po["total"],
        "invoice_total": inv["total"],
        "total_difference": total_diff,
        "total_outside_tolerance": abs(total_diff) > TOTAL_TOLERANCE_CAD,
        "discrepancies": discrepancies,
        "total_financial_impact": round(sum(d["financial_impact"] for d in discrepancies), 2),
    })
    result["status"] = "discrepancies" if (discrepancies or result["total_outside_tolerance"]) else "match"
    if result["total_financial_impact"] or discrepancies:
        result["approval_authority"] = approval_authority(result["total_financial_impact"])
    return result
