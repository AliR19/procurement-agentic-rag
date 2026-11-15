from langchain_core.messages import ToolMessage

from erp_api import main as erp_main

QUESTION = ("Can you check the status of purchase order 450012, compare it with the supplier invoice, "
            "and tell me whether there are any discrepancies?")


def _tools_used(result):
    return [m.name for m in result["messages"] if isinstance(m, ToolMessage)]


def test_po_investigation_uses_erp_and_rag_tools_and_cites_policy(run):
    r = run.say(QUESTION)
    used = _tools_used(r)
    for t in ("get_purchase_order", "get_invoice_details", "compare_po_and_invoice",
              "search_procurement_policy", "get_supplier_information"):
        assert t in used
    answer = r["messages"][-1].content
    assert "595.20" in answer and "Procurement manager" in answer
    assert "[POL-100" in answer  # grounded citation from retrieved policy
    assert "__interrupt__" not in r


def test_clean_po_needs_no_action(run):
    answer = run.say("Check PO 450001 against its invoice")["messages"][-1].content
    assert "matches the PO" in answer


def test_unknown_po_is_reported_not_invented(run):
    answer = run.say("Check PO 459999")["messages"][-1].content
    assert "could not find" in answer


def test_policy_question_without_po(run):
    answer = run.say("Who has to approve a discrepancy over CAD 5000?")["messages"][-1].content
    assert "[POL-300" in answer


def test_write_requires_approval_and_nothing_is_written_before_it(run):
    run.say(QUESTION)
    r = run.say("Yes, please create the exception report")
    assert r["__interrupt__"][0].value["tool"] == "create_exception_report"
    assert erp_main._EXCEPTION_REPORTS == []  # paused: nothing written yet


def test_approved_report_is_filed(run):
    run.say(QUESTION)
    run.say("Yes, please create the exception report")
    r = run.decide(True)
    assert len(erp_main._EXCEPTION_REPORTS) == 1
    assert erp_main._EXCEPTION_REPORTS[0]["po_number"] == "450012"
    assert "EXR-0001" in r["messages"][-1].content


def test_rejected_report_is_not_filed(run):
    run.say(QUESTION)
    run.say("Please file the exception report")
    r = run.decide(False)
    assert erp_main._EXCEPTION_REPORTS == []
    assert "not filed" in r["messages"][-1].content
