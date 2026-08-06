"""
Arize Phoenix — fully open-source, runs 100% locally, no signup or card at all.
Gives you a trace UI showing every LLM call, latency, and token usage for free,
which is what LangSmith would otherwise charge for at scale.

Import and call setup_observability() BEFORE creating the graph/LLM in api/main.py
or your notebook, so LangChain/LangGraph calls get auto-instrumented.
"""
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor


def setup_observability():
    session = px.launch_app()  # opens a local UI, default http://localhost:6006
    LangChainInstrumentor().instrument()
    print(f"Phoenix UI running at: {session.url}")
    return session
