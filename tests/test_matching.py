import pytest

from agent_service.matching import approval_authority, business_days_late, compare


def _compare(erp, po_number):
    return compare(erp.get(f"/purchase-orders/{po_number}"), erp.get("/invoices", po_number=po_number))


def test_showcase_po_has_price_and_quantity_discrepancies(erp):
    r = _compare(erp, "450012")
    kinds = {d["type"] for d in r["discrepancies"]}
    assert kinds == {"price_variance", "quantity_variance"}
    price = next(d for d in r["discrepancies"] if d["type"] == "price_variance")
    assert price["variance_pct"] == 12.0
    assert price["financial_impact"] == pytest.approx(595.20)
    assert r["approval_authority"] == "Procurement manager"
    assert r["total_outside_tolerance"] is True


def test_clean_po_matches(erp):
    assert _compare(erp, "450001")["status"] == "match"


def test_price_within_tolerance_is_not_flagged(erp):
    r = _compare(erp, "450030")  # +1% price, tolerance is 2%
    assert r["status"] == "match" and r["discrepancies"] == []


def test_late_delivery_flagged_beyond_grace_period(erp):
    r = _compare(erp, "450007")
    late = next(d for d in r["discrepancies"] if d["type"] == "late_delivery")
    assert late["business_days_late"] > 3


def test_duplicate_invoice_detected(erp):
    r = _compare(erp, "450021")
    assert any(d["type"] == "duplicate_invoice" for d in r["discrepancies"])
    assert r["approval_authority"] == "Procurement director"


def test_not_invoiced(erp):
    assert _compare(erp, "450044")["status"] == "not_invoiced"


@pytest.mark.parametrize("amount,who", [(0, "Accounts payable supervisor"), (499.99, "Accounts payable supervisor"),
                                        (500, "Procurement manager"), (4999.99, "Procurement manager"),
                                        (5000, "Procurement director")])
def test_approval_authority_boundaries(amount, who):
    assert approval_authority(amount) == who


def test_business_days_skip_weekends():
    # Fri 2026-07-31 -> Mon 2026-08-03 is one business day
    assert business_days_late("2026-07-31", "2026-08-03") == 1
    assert business_days_late("2026-07-31", "2026-07-31") == 0
    assert business_days_late("2026-07-31", None) == 0
