"""Guardrails: the agent must stay grounded even when the model misbehaves.

A scripted fake model plays the part of a weak LLM: it skips the policy search, cites a policy it
never retrieved, and misreports what happened after a report was filed. These are the failures seen
when running the agent on a real 8B model (llama3.1).
"""
import itertools
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command

from agent_service import guardrails as g
from agent_service.agent import build_graph
from agent_service.main import create_app
from agent_service.tools import build_tools
from erp_api import main as erp_main

QUESTION = "Can you check purchase order 450012 against its invoice?"
_ids = itertools.count(1)


def call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"call-{next(_ids)}", "type": "tool_call"}


def investigate(*extra: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=[call("get_purchase_order", po_number="450012"),
                                             call("get_invoice_details", po_number="450012"),
                                             call("compare_po_and_invoice", po_number="450012"), *extra])


def first_retrieved(messages) -> str:
    return sorted(g.retrieved_citations(messages))[0]


class ScriptedChatModel(BaseChatModel):
    steps: list[Any]
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-test"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        step = self.steps[self.cursor]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=step(messages) if callable(step) else step)])


@pytest.fixture()
def scripted(erp, retriever):
    def make(*steps):
        graph = build_graph(ScriptedChatModel(steps=list(steps)), build_tools(erp, retriever))
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 25}

        class Runner:
            def __init__(self):
                self.graph = graph

            def say(self, text):
                return graph.invoke({"messages": [HumanMessage(text)]}, cfg)

            def decide(self, approved):
                return graph.invoke(Command(resume={"approved": approved}), cfg)

        return Runner()
    return make


def tools_used(result):
    return [m.name for m in result["messages"] if isinstance(m, ToolMessage)]


SKIPPED_POLICY = AIMessage(content="DRAFT without the policy: this text must never reach the user.")


def draft_with_real_and_invented_citation(messages) -> AIMessage:
    real = first_retrieved(messages)
    return AIMessage(content=f"PO 450012 has a 12% price variance. {real} sets the tolerance. "
                             "See also [POL-999 9. Invented rule].")


# ---------------------------------------------------------------- forced policy search + citations
def test_policy_search_is_run_when_the_model_skips_it_and_fake_citation_is_removed(scripted):
    # step 2 is the model answering WITHOUT searching: it is discarded, the search runs, the model answers again
    r = scripted(investigate(), SKIPPED_POLICY, draft_with_real_and_invented_citation).say(QUESTION)
    assert "search_procurement_policy" in tools_used(r)  # the model never asked for it
    assert not any("must never reach the user" in str(m.content) for m in r["messages"])
    answer = r["messages"][-1].content
    assert "[POL-999" not in answer and "Invented rule" not in answer
    assert first_retrieved(r["messages"]) in answer  # the genuine citation is kept
    assert "did not match any retrieved policy text" in answer


def test_guardrail_activity_is_reported_over_http(scripted):
    run = scripted(investigate(), SKIPPED_POLICY, draft_with_real_and_invented_citation)
    client = TestClient(create_app(graph=run.graph, model_label="scripted", embedder_name="hashing"))
    body = client.post("/chat", json={"thread_id": "g1", "message": QUESTION}).json()
    assert body["guardrails"]["forced_policy_search"] is True
    assert body["guardrails"]["citations_removed"] == ["[POL-999 9. Invented rule]"]
    forced = [t for t in body["tool_trace"] if t["tool"] == "search_procurement_policy"]
    assert forced and forced[0].get("forced") is True


def test_model_that_searches_on_its_own_is_not_flagged(scripted):
    search = call("search_procurement_policy", query="unit price tolerance price increase")
    run = scripted(investigate(search), lambda m: AIMessage(content=f"Price variance. {first_retrieved(m)}"))
    client = TestClient(create_app(graph=run.graph, model_label="scripted", embedder_name="hashing"))
    body = client.post("/chat", json={"thread_id": "g2", "message": QUESTION}).json()
    assert body["guardrails"] == {"forced_policy_search": False, "citations_removed": [], "outcome_replaced": False,
                                  "sources_appended": False}
    assert not any(t.get("forced") for t in body["tool_trace"])


def test_clean_match_does_not_force_a_policy_search(scripted):
    ok = AIMessage(content="", tool_calls=[call("get_purchase_order", po_number="450001"),
                                           call("get_invoice_details", po_number="450001"),
                                           call("compare_po_and_invoice", po_number="450001")])
    r = scripted(ok, AIMessage(content="PO 450001 matches its invoice.")).say("Check PO 450001")
    assert "search_procurement_policy" not in tools_used(r)


def test_sources_are_listed_when_the_model_searched_but_cited_nothing(scripted):
    run = scripted(investigate(), SKIPPED_POLICY, AIMessage(content="Price is 12% above the PO, outside tolerance."))
    client = TestClient(create_app(graph=run.graph, model_label="scripted", embedder_name="hashing"))
    body = client.post("/chat", json={"thread_id": "g3", "message": QUESTION}).json()
    assert body["guardrails"]["sources_appended"] is True
    head, listed = body["reply"].split("Policy passages retrieved for this answer: ")
    assert head.startswith("Price is 12% above the PO")
    cites = listed.split("; ")
    assert 1 <= len(cites) <= 3 and all(c.startswith("[POL-") or c.startswith("[AGR-") for c in cites)
    retrieved = g.retrieved_citations(run.graph.get_state({"configurable": {"thread_id": "g3"}}).values["messages"])
    assert set(cites) <= retrieved  # only passages the search really returned


def test_no_sources_line_when_the_policy_was_not_searched(scripted):
    ok = AIMessage(content="", tool_calls=[call("get_purchase_order", po_number="450001"),
                                           call("get_invoice_details", po_number="450001"),
                                           call("compare_po_and_invoice", po_number="450001")])
    r = scripted(ok, AIMessage(content="PO 450001 matches its invoice.")).say("Check PO 450001")
    assert r["messages"][-1].content == "PO 450001 matches its invoice."


# ---------------------------------------------------------------- outcome of the write
def _until_approval(scripted, final_message):
    run = scripted(
        investigate(), SKIPPED_POLICY,
        AIMessage(content="Discrepancies found. Shall I file an exception report?"),
        AIMessage(content="", tool_calls=[call("create_exception_report", po_number="450012",
                                               invoice_number="INV-90012",
                                               summary="Price and quantity variance on PO 450012",
                                               discrepancies=[])]),
        final_message)
    run.say(QUESTION)
    paused = run.say("Yes, please file the exception report")
    assert paused["__interrupt__"][0].value["tool"] == "create_exception_report"
    assert erp_main._EXCEPTION_REPORTS == []
    return run


def test_wrong_status_after_approval_is_replaced_with_the_truth(scripted):
    run = _until_approval(scripted, AIMessage(content="The report is waiting for approval."))
    answer = run.decide(True)["messages"][-1].content
    assert "EXR-0001" in answer and "waiting" not in answer
    assert len(erp_main._EXCEPTION_REPORTS) == 1


def test_correct_message_after_approval_is_kept_as_written(scripted):
    run = _until_approval(scripted, AIMessage(content="Done: it was filed as EXR-0001."))
    assert run.decide(True)["messages"][-1].content == "Done: it was filed as EXR-0001."


def test_wrong_status_after_rejection_is_replaced_with_the_truth(scripted):
    run = _until_approval(scripted, AIMessage(content="All done, the report is on its way!"))
    answer = run.decide(False)["messages"][-1].content
    assert "not filed" in answer and "rejected" in answer
    assert erp_main._EXCEPTION_REPORTS == []


# ---------------------------------------------------------------- pure functions
ALLOWED = {"[POL-100 2. Matching tolerances]", "[POL-300 2. Contents]"}


def test_verify_citations_keeps_retrieved_and_removes_the_rest():
    text, removed = g.verify_citations(
        "Tolerance is 2% [POL-100 2. Matching tolerances] and (see [POL-777 1. Made up]).", ALLOWED)
    assert "[POL-100 2. Matching tolerances]" in text and "POL-777" not in text and "()" not in text
    assert removed == ["[POL-777 1. Made up]"]


def test_verify_citations_accepts_document_id_only_if_a_section_was_retrieved():
    assert g.verify_citations("See [POL-100].", ALLOWED)[1] == []
    assert g.verify_citations("See [POL-400].", ALLOWED)[1] == ["[POL-400]"]


def test_verify_citations_leaves_text_without_citations_untouched():
    assert g.verify_citations("Nothing to cite here.", ALLOWED) == ("Nothing to cite here.", [])


def test_missing_policy_search_only_when_discrepancies_and_no_search_this_turn():
    compare = lambda body: ToolMessage(content=body, name="compare_po_and_invoice", tool_call_id="x")
    bad = '{"discrepancies": [{"type": "price_variance"}], "status": "mismatch"}'
    assert "price tolerance" in g.missing_policy_search([HumanMessage("q"), compare(bad)])
    assert g.missing_policy_search([HumanMessage("q"), compare('{"discrepancies": [], "status": "match"}')]) is None
    search = ToolMessage(content="[]", name="search_procurement_policy", tool_call_id="y")
    assert g.missing_policy_search([HumanMessage("q"), compare(bad), search]) is None
    # a comparison from an EARLIER turn does not count
    assert g.missing_policy_search([HumanMessage("q"), compare(bad), HumanMessage("thanks")]) is None
