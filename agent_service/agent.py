"""LangGraph agent with a human-approval gate.

    START -> agent --(tool calls)--> tools -> approval -> agent -> ... -> END
                  \\--(no tool calls)--------------------------------> END

* `agent`    : the LLM decides which tools to call (or answers).
* `tools`    : runs read-only tools immediately.
* `approval` : pauses the graph with `interrupt()` before any tool that WRITES to the ERP.
               The run resumes only when a human approves or rejects.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from .tools import SENSITIVE_TOOLS

SYSTEM_PROMPT = """You are a procurement assistant for an accounts-payable and purchasing team.

Rules:
- Use tools to get facts. Never invent purchase orders, invoices, amounts or policy text.
- For PO/invoice questions: fetch the PO and invoice, run compare_po_and_invoice (do not do the
  arithmetic yourself), and search_procurement_policy for the relevant rules.
- Cite policy statements using the citation strings returned by search_procurement_policy,
  for example [POL-100 2. Matching tolerances].
- Explain discrepancies clearly with amounts in CAD, who must approve according to policy, and offer
  to file an exception report. Only call create_exception_report after the user asks for it.
  Filing requires human approval, so tell the user it is waiting for approval.
- If data is missing, say so plainly."""


def _last_tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return list(m.tool_calls or [])
    return []


def build_graph(llm: Any, tools: list[Any], checkpointer: Any | None = None):
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def execute(call: dict[str, Any]) -> ToolMessage:
        tool = tool_map.get(call["name"])
        try:
            content = tool.invoke(call["args"]) if tool else json.dumps({"error": f"Unknown tool {call['name']}"})
        except Exception as exc:  # surface tool failures to the model instead of crashing the run
            content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        return ToolMessage(content=content, name=call["name"], tool_call_id=call["id"])

    def agent(state: MessagesState) -> dict[str, Any]:
        reply = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [reply]}

    def run_safe_tools(state: MessagesState) -> dict[str, Any]:
        calls = [c for c in _last_tool_calls(state["messages"]) if c["name"] not in SENSITIVE_TOOLS]
        return {"messages": [execute(c) for c in calls]}

    def approval_gate(state: MessagesState) -> dict[str, Any]:
        out: list[ToolMessage] = []
        for call in _last_tool_calls(state["messages"]):
            if call["name"] not in SENSITIVE_TOOLS:
                continue
            decision = interrupt({
                "tool": call["name"],
                "args": call["args"],
                "message": "This action writes to the ERP. Approve to proceed.",
            })
            if isinstance(decision, dict) and decision.get("approved"):
                out.append(execute(call))
            else:
                out.append(ToolMessage(
                    content="REJECTED: a human reviewer declined this action. Nothing was written.",
                    name=call["name"], tool_call_id=call["id"]))
        return {"messages": out}

    def after_agent(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", run_safe_tools)
    g.add_node("approval", approval_gate)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, {"tools": "tools", END: END})
    g.add_edge("tools", "approval")
    g.add_edge("approval", "agent")
    return g.compile(checkpointer=checkpointer or MemorySaver())
