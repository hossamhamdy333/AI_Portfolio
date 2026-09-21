"""
The assistant ends knowledge-base answers with a "Source: ..." line. These
tests cover pulling that line out of the answer (so the page can show it as
"From:" badges) and that retrieval tells the model each document's filename.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")
os.environ.setdefault("QDRANT_URL", "http://test:6333")
os.environ.setdefault("QDRANT_API_KEY", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_sources.db")

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools.retriever import create_retriever_tool

import agent
import main
from agent import split_sources
from database import init_db

init_db()


@pytest.mark.parametrize("text, answer, sources", [
    ("He is an engineer.\n\nSource: Hossam_Hamdy_CV.pdf", "He is an engineer.", ["Hossam_Hamdy_CV.pdf"]),
    ("He is an engineer.\n\n(Source: Hossam Hamdy Fakry's Resume)", "He is an engineer.", ["Hossam Hamdy Fakry's Resume"]),
    ("Answer.\n**Sources:** a.pdf, b.txt", "Answer.", ["a.pdf", "b.txt"]),
    ("Answer.\nSources: a.pdf and b.pdf.", "Answer.", ["a.pdf", "b.pdf"]),
    ("Answer.\nSource: a.pdf; b.md", "Answer.", ["a.pdf", "b.md"]),
    ("Answer.\n*Source: notes.txt*", "Answer.", ["notes.txt"]),
    ("Answer.\nSOURCE: notes.txt   \n\n", "Answer.", ["notes.txt"]),
    ("Line one.\nLine two.\nSource: a.pdf", "Line one.\nLine two.", ["a.pdf"]),
])
def test_source_line_is_pulled_out_of_the_answer(text, answer, sources):
    assert split_sources(text) == (answer, sources)


@pytest.mark.parametrize("text", [
    "No citation here.",
    "The source: code lives in the repo.\nThat is all.",      # the word appears, but not as the last line
    "Read the Source: section below.\nThen continue.",
    "",
])
def test_text_without_a_source_line_is_unchanged(text):
    assert split_sources(text) == (text, [])


def test_a_reply_that_is_only_a_citation_is_kept_as_the_answer():
    assert split_sources("Source: a.pdf") == ("Source: a.pdf", [])


@pytest.mark.parametrize("placeholder", ["None", "N/A", "unknown"])
def test_placeholder_sources_are_dropped_but_the_line_is_still_removed(placeholder):
    assert split_sources(f"Answer.\nSource: {placeholder}") == ("Answer.", [])


def test_duplicates_are_removed_and_the_list_is_capped():
    names = ", ".join(f"doc{i}.pdf" for i in range(9))
    _, sources = split_sources(f"Answer.\nSources: a.pdf, a.pdf, {names}")
    assert sources[0] == "a.pdf" and len(sources) == agent.MAX_SOURCES and len(set(sources)) == len(sources)


def test_very_long_names_are_shortened():
    _, sources = split_sources("Answer.\nSource: " + "x" * 300 + ".pdf")
    assert len(sources[0]) == 80


# ------------------------------------------------------------ endpoint

def _chat(monkeypatch, reply, email):
    monkeypatch.setattr(main, "run_agent", lambda q, user_id: reply)
    client = TestClient(main.app)
    token = client.post("/auth/register", json={"email": email, "password": "a-real-password-123"}).json()["access_token"]
    return client.post("/chat", json={"query": "who is he?"}, headers={"Authorization": "Bearer " + token})


def test_chat_returns_the_sources_separately(monkeypatch):
    resp = _chat(monkeypatch, "He is an engineer.\n\nSource: Hossam_Hamdy_CV.pdf", "src1@example.com")
    assert resp.status_code == 200
    assert resp.json() == {"answer": "He is an engineer.", "sources": ["Hossam_Hamdy_CV.pdf"], "blocked": False}


def test_chat_without_a_source_line_returns_an_empty_list(monkeypatch):
    resp = _chat(monkeypatch, "55 * 3 = 165", "src2@example.com")
    assert resp.json() == {"answer": "55 * 3 = 165", "sources": [], "blocked": False}


# ------------------------------------------------------------ retrieval formatting

class _FixedRetriever(BaseRetriever):
    docs: list

    def _get_relevant_documents(self, query, *, run_manager):
        return [d.model_copy(deep=True) for d in self.docs]


def _tool_output(docs):
    tool = create_retriever_tool(
        agent.SourceTaggedRetriever(inner=_FixedRetriever(docs=docs)),
        "kb", "search", document_prompt=agent.SOURCE_DOCUMENT_PROMPT,
    )
    return tool.invoke({"query": "refund"})


def test_retrieved_chunks_show_the_model_their_real_document_name():
    out = _tool_output([LCDocument(page_content="Refunds take 30 days.", metadata={"source": "policy.pdf", "user_id": 1})])
    assert out == "[Document: policy.pdf]\nRefunds take 30 days."


def test_a_chunk_without_source_metadata_does_not_break_retrieval():
    out = _tool_output([
        LCDocument(page_content="Old chunk.", metadata={"user_id": 1}),
        LCDocument(page_content="New chunk.", metadata={"source": "new.txt"}),
    ])
    assert "[Document: unknown document]\nOld chunk." in out
    assert "[Document: new.txt]\nNew chunk." in out   # a real name is never overwritten


def test_build_agent_wraps_the_retriever_and_passes_the_document_prompt(monkeypatch):
    seen = {}
    inner = object.__new__(_FixedRetriever)

    class _VS:
        def as_retriever(self, **kwargs):
            return _FixedRetriever(docs=[])

    monkeypatch.setattr(agent, "_vectorstore", _VS())
    monkeypatch.setattr(agent, "get_llm", lambda: object())
    monkeypatch.setattr(agent, "create_retriever_tool", lambda *a, **k: seen.update(args=a, kwargs=k) or "tool")
    monkeypatch.setattr(agent, "create_agent", lambda **k: "agent")
    assert agent.build_agent(1) == "agent"
    assert isinstance(seen["args"][0], agent.SourceTaggedRetriever)
    assert seen["kwargs"]["document_prompt"] is agent.SOURCE_DOCUMENT_PROMPT
