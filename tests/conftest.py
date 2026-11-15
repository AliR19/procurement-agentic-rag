import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent_service.agent import build_graph
from agent_service.demo_llm import DemoChatModel
from agent_service.rag.embeddings import HashingEmbedder
from agent_service.rag.hybrid import HybridRetriever
from agent_service.tools import ErpClient, build_tools
from erp_api import main as erp_main

POLICY_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"


@pytest.fixture(autouse=True)
def clean_erp_reports():
    erp_main._EXCEPTION_REPORTS.clear()
    yield
    erp_main._EXCEPTION_REPORTS.clear()


@pytest.fixture(scope="session")
def retriever():
    return HybridRetriever.from_directory(POLICY_DIR, HashingEmbedder())


@pytest.fixture()
def erp():
    return ErpClient(client=TestClient(erp_main.app, base_url="http://testserver"), api_key="dev-key")


@pytest.fixture()
def graph(erp, retriever):
    return build_graph(DemoChatModel(), build_tools(erp, retriever))


@pytest.fixture()
def run(graph):
    """Helper: run(message) / approve(bool) on a fresh thread."""
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 25}

    class Runner:
        def say(self, text):
            return graph.invoke({"messages": [HumanMessage(text)]}, cfg)

        def decide(self, approved):
            return graph.invoke(Command(resume={"approved": approved}), cfg)

    return Runner()
