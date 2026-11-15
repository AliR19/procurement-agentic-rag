"""Generate SIMULATED procurement data for the demo ERP API.

Everything produced here is synthetic: fictional suppliers, made-up purchase orders and
invoices. A fixed random seed makes the output reproducible. A few invoices are given
deliberate discrepancies (price, quantity, late delivery, duplicate) so the agent has
something real to find.

Run:  python scripts/generate_data.py
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
OUT = Path(__file__).resolve().parent.parent / "erp_api" / "seed.json"

SUPPLIERS = [
    ("S1001", "Northwind Industrial Supply", "Canada", "Net 30", 4.6),
    ("S1002", "Maple Ridge Components", "Canada", "Net 45", 4.2),
    ("S1003", "Harbourline Packaging", "Canada", "Net 30", 3.9),
    ("S1004", "Cedar Peak Electronics", "United States", "Net 30", 4.4),
    ("S1005", "Atlantic Fastener Co.", "Canada", "Net 60", 4.0),
    ("S1006", "Lakeshore Office Systems", "Canada", "Net 30", 4.7),
    ("S1007", "Redwood Safety Gear", "United States", "Net 45", 3.8),
    ("S1008", "Tidewater Logistics Parts", "Canada", "Net 30", 4.1),
]

CATALOG = [
    ("SKU-1001", "Hex bolt M10 (box of 100)", 18.50),
    ("SKU-1002", "Industrial pallet wrap 20in", 24.00),
    ("SKU-1003", "Laptop docking station", 189.00),
    ("SKU-1004", "Safety gloves (pack of 12)", 32.75),
    ("SKU-1005", "Corrugated shipping box 18x12x10", 1.85),
    ("SKU-1006", "LED panel light 60W", 64.20),
    ("SKU-1007", "Ergonomic office chair", 245.00),
    ("SKU-1008", "Barcode scanner (wireless)", 129.00),
    ("SKU-1009", "Steel shelving unit", 310.00),
    ("SKU-1010", "Printer toner cartridge", 87.40),
]

PO_START = 450001
PO_COUNT = 50


def business_days_after(start: date, days: int) -> date:
    d = start
    added = 0
    while added < days:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


def main() -> None:
    rng = random.Random(SEED)
    suppliers = [
        {
            "supplier_id": sid,
            "name": name,
            "country": country,
            "payment_terms": terms,
            "rating": rating,
            "contact_email": f"accounts@{name.split()[0].lower()}.example.com",
            "contract_id": f"CTR-{sid[1:]}",
        }
        for sid, name, country, terms, rating in SUPPLIERS
    ]

    purchase_orders, invoices = [], []
    base = date(2026, 6, 1)

    for i in range(PO_COUNT):
        po_number = str(PO_START + i)
        supplier = rng.choice(suppliers)
        created = base + timedelta(days=rng.randint(0, 120))
        expected = business_days_after(created, rng.randint(5, 15))
        lines = []
        for ln in range(1, rng.randint(1, 3) + 1):
            sku, desc, price = rng.choice(CATALOG)
            lines.append(
                {
                    "line": ln,
                    "sku": sku,
                    "description": desc,
                    "quantity": rng.choice([10, 20, 25, 40, 50, 100]),
                    "unit_price": price,
                }
            )
        total = round(sum(l["quantity"] * l["unit_price"] for l in lines), 2)
        po = {
            "po_number": po_number,
            "supplier_id": supplier["supplier_id"],
            "status": "Received",
            "currency": "CAD",
            "created_date": created.isoformat(),
            "expected_delivery": expected.isoformat(),
            "actual_delivery": expected.isoformat(),
            "lines": lines,
            "total": total,
        }
        purchase_orders.append(po)

    # Decide scenarios. 450012 is the showcase case used in the README / demo.
    scenario = {p["po_number"]: "clean" for p in purchase_orders}
    scenario["450012"] = "price_and_quantity"
    scenario["450007"] = "late_delivery"
    scenario["450021"] = "duplicate_invoice"
    scenario["450030"] = "price_minor"  # within tolerance
    scenario["450044"] = "not_invoiced"
    for po_number in ("450003", "450015", "450033", "450040"):
        scenario[po_number] = "price_major"
    for po_number in ("450018", "450027"):
        scenario[po_number] = "quantity_short"

    inv_counter = 90000
    for po in purchase_orders:
        kind = scenario[po["po_number"]]
        if kind == "not_invoiced":
            po["status"] = "Open - awaiting invoice"
            continue

        inv_lines = []
        for l in po["lines"]:
            qty, price = l["quantity"], l["unit_price"]
            if kind in ("price_and_quantity", "price_major") and l["line"] == 1:
                price = round(price * (1.12 if kind == "price_and_quantity" else 1.08), 2)
            if kind == "price_minor" and l["line"] == 1:
                price = round(price * 1.01, 2)
            if kind in ("price_and_quantity", "quantity_short") and l["line"] == 1:
                qty = int(qty * 0.8)
            inv_lines.append(
                {"line": l["line"], "sku": l["sku"], "quantity": qty, "unit_price": price}
            )

        if kind == "late_delivery":
            exp = date.fromisoformat(po["expected_delivery"])
            po["actual_delivery"] = business_days_after(exp, 6).isoformat()

        inv_counter += 1
        inv_date = date.fromisoformat(po["actual_delivery"]) + timedelta(days=2)
        invoice = {
            "invoice_number": f"INV-{inv_counter}",
            "po_number": po["po_number"],
            "supplier_id": po["supplier_id"],
            "invoice_date": inv_date.isoformat(),
            "currency": "CAD",
            "lines": inv_lines,
            "total": round(sum(x["quantity"] * x["unit_price"] for x in inv_lines), 2),
        }
        invoices.append(invoice)

        if kind == "duplicate_invoice":
            inv_counter += 1
            dup = dict(invoice)
            dup["invoice_number"] = f"INV-{inv_counter}"
            dup["invoice_date"] = (inv_date + timedelta(days=5)).isoformat()
            invoices.append(dup)

    data = {
        "_note": "SIMULATED DATA - fictional suppliers and documents for a portfolio demo.",
        "suppliers": suppliers,
        "purchase_orders": purchase_orders,
        "invoices": invoices,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2))
    print(f"Wrote {OUT} ({len(purchase_orders)} POs, {len(invoices)} invoices, {len(suppliers)} suppliers)")


if __name__ == "__main__":
    main()
