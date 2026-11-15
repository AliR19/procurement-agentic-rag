"""LLM provider selection. Set LLM_PROVIDER to one of:

  demo    - rule-based stand-in, no model needed (default, for tests / offline demos)
  ollama  - local model via Ollama. Use a tool-calling model, e.g. llama3.1 or llama3.2
  openai  - OpenAI API (needs OPENAI_API_KEY)
"""
from __future__ import annotations

import os
from typing import Any


def get_llm() -> tuple[Any, str]:
    provider = os.getenv("LLM_PROVIDER", "demo").lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "llama3.1")
        llm = ChatOllama(model=model, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                         temperature=0)
        return llm, f"ollama:{model}"
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0), f"openai:{model}"
    if provider == "demo":
        from .demo_llm import DemoChatModel

        return DemoChatModel(), "demo (rule-based, no LLM)"
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use demo, ollama or openai.")
