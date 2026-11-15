from fastapi.testclient import TestClient

from erp_api.main import app

client = TestClient(app)
H = {"x-api-key": "dev-key"}


def test_requires_api_key():
    assert client.get("/purchase-orders/450012").status_code == 401
    assert client.get("/purchase-orders/450012", headers={"x-api-key": "wrong"}).status_code == 401


def test_get_po_and_supplier_and_invoices():
    po = client.get("/purchase-orders/450012", headers=H).json()
    assert po["supplier_id"] == "S1008"
    assert client.get("/suppliers/S1008", headers=H).json()["name"].startswith("Tidewater")
    assert client.get("/invoices", params={"po_number": "450012"}, headers=H).json()[0]["invoice_number"] == "INV-90012"


def test_unknown_po_is_404():
    assert client.get("/purchase-orders/999999", headers=H).status_code == 404


def test_create_and_list_exception_report():
    r = client.post("/exception-reports", headers=H, json={"po_number": "450012", "summary": "Price variance"})
    assert r.status_code == 201 and r.json()["report_id"] == "EXR-0001"
    assert len(client.get("/exception-reports", headers=H).json()) == 1


def test_exception_report_validates_input():
    assert client.post("/exception-reports", headers=H, json={"po_number": "450012", "summary": "x"}).status_code == 422
