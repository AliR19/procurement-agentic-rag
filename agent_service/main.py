"""Agent service: chat API + web UI for the procurement assistant."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from .agent import build_graph
from .config import CHROMA_DIR, POLICY_DIR
from .llm import get_llm
from .rag.embeddings import get_embedder
from .rag.hybrid import HybridRetriever
from .tools import ErpClient, build_tools

log = logging.getLogger("agent_service")
STATIC = Path(__file__).resolve().parent / "static"


class ChatIn(BaseModel):
    thread_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=2000)


class ApproveIn(BaseModel):
    thread_id: str = Field(min_length=1, max_length=64)
    approved: bool


def create_app(graph: Any | None = None, model_label: str = "custom", embedder_name: str = "custom") -> FastAPI:
    state: dict[str, Any] = {"graph": graph, "model": model_label, "embedder": embedder_name}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["graph"] is None:
            llm, label = get_llm()
            embedder = get_embedder()
            retriever = HybridRetriever.from_directory(POLICY_DIR, embedder, persist_dir=CHROMA_DIR)
            state.update(graph=build_graph(llm, build_tools(ErpClient(), retriever)),
                         model=label, embedder=embedder.name, chunks=len(retriever.chunks))
            log.info("Ready: model=%s embeddings=%s chunks=%s", label, embedder.name, len(retriever.chunks))
        yield

    app = FastAPI(title="Procurement Agentic RAG Assistant", version="1.0.0", lifespan=lifespan)

    def _trace(messages: list[Any]) -> list[dict[str, Any]]:
        """Tool calls made since the last user message, for display in the UI."""
        start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
        calls: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for m in messages[start + 1:]:
            if isinstance(m, AIMessage):
                forced = bool(m.additional_kwargs.get("guardrail"))
                for c in m.tool_calls or []:
                    calls[c["id"]] = {"tool": c["name"], "args": c["args"], "result": None}
                    if forced:
                        calls[c["id"]]["forced"] = True  # run by a guardrail, not requested by the model
                    order.append(c["id"])
            elif isinstance(m, ToolMessage) and m.tool_call_id in calls:
                text = m.content if isinstance(m.content, str) else json.dumps(m.content)
                calls[m.tool_call_id]["result"] = text[:600] + ("..." if len(text) > 600 else "")
        return [calls[i] for i in order]

    def _respond(result: dict[str, Any]) -> dict[str, Any]:
        msgs = result["messages"]
        pending = None
        interrupts = result.get("__interrupt__")
        if interrupts:
            pending = interrupts[0].value
        replies = [m for m in msgs if isinstance(m, AIMessage) and m.content]
        start = max((i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)), default=-1)
        turn = msgs[start + 1:]
        turn_replies = [m for m in turn if isinstance(m, AIMessage) and m.content]
        info = turn_replies[-1].additional_kwargs.get("guardrails", {}) if turn_replies else {}
        return {
            "reply": (replies[-1].content if replies else ""),
            "pending_approval": pending,
            "tool_trace": _trace(msgs),
            "model": state["model"],
            # What the guardrails did this turn (see guardrails.py): useful to see how much the model needed them.
            "guardrails": {
                "forced_policy_search": any(isinstance(m, AIMessage) and m.additional_kwargs.get("guardrail")
                                            for m in turn),
                "citations_removed": info.get("citations_removed", []),
                "outcome_replaced": info.get("outcome_replaced", False),
            },
        }

    cfg = lambda tid: {"configurable": {"thread_id": tid}, "recursion_limit": 25}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": state["model"], "embeddings": state["embedder"]}

    @app.post("/chat")
    def chat(body: ChatIn) -> dict[str, Any]:
        try:
            result = state["graph"].invoke({"messages": [HumanMessage(body.message)]}, cfg(body.thread_id))
        except Exception as exc:
            log.exception("chat failed")
            raise HTTPException(502, f"Agent run failed: {type(exc).__name__}: {exc}") from exc
        return _respond(result)

    @app.post("/approve")
    def approve(body: ApproveIn) -> dict[str, Any]:
        snapshot = state["graph"].get_state(cfg(body.thread_id))
        if not any(t.interrupts for t in snapshot.tasks):
            raise HTTPException(409, "Nothing is awaiting approval for this thread.")
        try:
            result = state["graph"].invoke(Command(resume={"approved": body.approved}), cfg(body.thread_id))
        except Exception as exc:
            log.exception("approve failed")
            raise HTTPException(502, f"Agent run failed: {type(exc).__name__}: {exc}") from exc
        return _respond(result)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
