from fastapi.testclient import TestClient

from agent_service.main import create_app
from erp_api import main as erp_main


def _client(graph):
    return TestClient(create_app(graph=graph, model_label="demo-test", embedder_name="hashing"))


def test_health(graph):
    body = _client(graph).get("/health").json()
    assert body["status"] == "ok" and body["model"] == "demo-test"


def test_chat_returns_reply_and_tool_trace(graph):
    r = _client(graph).post("/chat", json={"thread_id": "a", "message": "Check PO 450012"}).json()
    assert "595.20" in r["reply"]
    assert [t["tool"] for t in r["tool_trace"]][:3] == ["get_purchase_order", "get_invoice_details", "compare_po_and_invoice"]
    assert r["pending_approval"] is None


def test_approval_flow_over_http(graph):
    c = _client(graph)
    c.post("/chat", json={"thread_id": "b", "message": "Check PO 450012"})
    pending = c.post("/chat", json={"thread_id": "b", "message": "yes create the exception report"}).json()
    assert pending["pending_approval"]["tool"] == "create_exception_report"
    assert erp_main._EXCEPTION_REPORTS == []
    done = c.post("/approve", json={"thread_id": "b", "approved": True}).json()
    assert "EXR-0001" in done["reply"] and done["pending_approval"] is None
    assert len(erp_main._EXCEPTION_REPORTS) == 1


def test_approve_without_pending_request_is_409(graph):
    c = _client(graph)
    c.post("/chat", json={"thread_id": "c", "message": "Check PO 450001"})
    assert c.post("/approve", json={"thread_id": "c", "approved": True}).status_code == 409


def test_input_validation(graph):
    c = _client(graph)
    assert c.post("/chat", json={"thread_id": "d", "message": ""}).status_code == 422
    assert c.get("/").status_code == 200  # UI served
