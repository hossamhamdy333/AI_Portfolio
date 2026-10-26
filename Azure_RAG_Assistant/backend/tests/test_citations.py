"""
Page numbers: every PDF chunk is stored with its page, the model sees each
chunk as "[n] file, page p", and the /chat answer comes back with citations
that say which file and page each [n] points to.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")
os.environ.setdefault("QDRANT_URL", "http://test:6333")
os.environ.setdefault("QDRANT_API_KEY", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_citations.db")

from fastapi.testclient import TestClient
from langchain_core.documents import Document as LCDocument
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

import agent
import main
import text_processing
from citations import build_citations, source_labels
from database import init_db

init_db()


# ------------------------------------------------------------ ingestion

class _FakeTextPage:
    def __init__(self, text):
        self._text = text

    def get_text_range(self):
        return self._text


class _FakePage:
    def __init__(self, text):
        self._text = text

    def get_textpage(self):
        return _FakeTextPage(self._text)


class _FakePdf:
    def __init__(self, pages):
        self._pages = [_FakePage(t or "") for t in pages]

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, index):
        return self._pages[index]

    def close(self):
        pass


def _fake_pdf(monkeypatch, pages):
    monkeypatch.setattr(text_processing.pdfium, "PdfDocument", lambda _: _FakePdf(pages))


def test_pdf_pages_keep_their_real_numbers_and_blank_pages_are_skipped(monkeypatch):
    _fake_pdf(monkeypatch, ["first page", None, "third page"])
    assert text_processing.extract_pages(b"x", "a.pdf") == [(1, "first page"), (3, "third page")]


def test_text_files_have_no_page_number():
    assert text_processing.extract_pages(b"hello", "a.txt") == [(None, "hello")]


def test_extract_text_still_returns_one_string(monkeypatch):
    _fake_pdf(monkeypatch, ["one", "two"])
    assert text_processing.extract_text(b"x", "a.pdf") == "one\ntwo\n"


def test_every_chunk_is_stored_with_its_page(monkeypatch):
    _fake_pdf(monkeypatch, ["Alpha. " * 300, "Beta. " * 300])
    stored = []
    monkeypatch.setattr(text_processing.QdrantVectorStore, "from_documents", lambda docs, *a, **k: stored.extend(docs))

    count = text_processing.process_and_upsert(b"x", "report.pdf", user_id=1, document_id=7)

    assert count == len(stored) > 2
    for doc in stored:
        assert doc.metadata["source"] == "report.pdf" and doc.metadata["document_id"] == 7
        # a chunk never mixes two pages
        assert ("Alpha" in doc.page_content) == (doc.metadata["page"] == 1)
        assert ("Beta" in doc.page_content) == (doc.metadata["page"] == 2)
    assert {d.metadata["page"] for d in stored} == {1, 2}


def test_a_text_file_chunk_has_no_page_key(monkeypatch):
    stored = []
    monkeypatch.setattr(text_processing.QdrantVectorStore, "from_documents", lambda docs, *a, **k: stored.extend(docs))
    text_processing.process_and_upsert(b"some notes", "notes.txt", user_id=1, document_id=1)
    assert "page" not in stored[0].metadata


# ------------------------------------------------------------ build_citations

CHUNKS = [
    {"ref": 1, "source": "policy.pdf", "page": 3, "text": "Refunds take 30 days."},
    {"ref": 2, "source": "notes.txt", "page": None, "text": "Shipping is free over 50 dollars."},
]


def test_numbers_become_citations_with_file_and_page():
    answer, citations, invalid = build_citations("Refunds take 30 days. [1] Shipping is free. [2]", CHUNKS)
    assert answer == "Refunds take 30 days. [1] Shipping is free. [2]"
    assert invalid == []
    assert [(c["ref"], c["source"], c["page"], c["label"]) for c in citations] == [
        (1, "policy.pdf", 3, "policy.pdf p.3"),
        (2, "notes.txt", None, "notes.txt"),
    ]
    assert citations[0]["snippet"] == "Refunds take 30 days."
    assert source_labels(citations) == ["policy.pdf p.3", "notes.txt"]


def test_a_repeated_number_gives_one_citation():
    _, citations, _ = build_citations("A [1]. B [1].", CHUNKS)
    assert len(citations) == 1


def test_a_made_up_number_is_removed_and_reported():
    answer, citations, invalid = build_citations("Refunds take 30 days [1] and more [9].", CHUNKS)
    assert answer == "Refunds take 30 days [1] and more."
    assert [c["ref"] for c in citations] == [1]
    assert invalid == [9]


def test_no_numbers_means_no_citations():
    assert build_citations("55 * 3 = 165", CHUNKS) == ("55 * 3 = 165", [], [])


def test_the_snippet_is_capped():
    chunks = [{"ref": 1, "source": "a.pdf", "page": 1, "text": "x" * 900}]
    _, citations, _ = build_citations("Yes [1]", chunks)
    assert len(citations[0]["snippet"]) == 200


# ------------------------------------------------------------ retrieval + endpoint

class _FixedRetriever(BaseRetriever):
    docs: list

    def _get_relevant_documents(self, query, *, run_manager):
        return [d.model_copy(deep=True) for d in self.docs]


class _FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _fake_model():
    call = AIMessage(content="", tool_calls=[{"name": "company_knowledge_base", "args": {"query": "refund"}, "id": "c1"}])
    final = AIMessage(content="Refunds take 30 days. [1]")
    return _FakeToolModel(messages=iter([call, final]))


class _VS:
    def as_retriever(self, **kwargs):
        doc = LCDocument(page_content="Refunds take 30 days.", metadata={"source": "policy.pdf", "page": 3, "user_id": 1})
        return _FixedRetriever(docs=[doc])


def _plain_model(text="Refunds take 30 days. [1]"):
    return lambda: _FakeToolModel(messages=iter([AIMessage(content=text)]))


def test_every_question_searches_first_and_records_the_chunks(monkeypatch):
    monkeypatch.setattr(agent, "_vectorstore", _VS())
    monkeypatch.setattr(agent, "get_llm", _plain_model())

    answer, chunks = agent.run_agent_cited("refund policy?", user_id=1)

    assert answer == "Refunds take 30 days. [1]"
    assert chunks == [{"ref": 1, "source": "policy.pdf", "page": 3, "text": "Refunds take 30 days."}]


def test_no_passages_means_no_model_call(monkeypatch):
    class _Empty:
        def as_retriever(self, **kwargs):
            return _FixedRetriever(docs=[])

    def boom():
        raise AssertionError("the model must not be called without passages")

    monkeypatch.setattr(agent, "_vectorstore", _Empty())
    monkeypatch.setattr(agent, "get_llm", boom)
    assert agent.run_agent_cited("what is a loss function?", user_id=1) == (agent.NOT_FOUND, [])


def test_the_passages_and_question_reach_the_model(monkeypatch):
    sent = {}

    class _Spy:
        def invoke(self, messages):
            sent["messages"] = messages
            return AIMessage(content="ok [1]")

    monkeypatch.setattr(agent, "_vectorstore", _VS())
    monkeypatch.setattr(agent, "get_llm", lambda: _Spy())
    agent.run_agent_cited("refund policy?", user_id=1)
    assert "[1] policy.pdf, page 3\nRefunds take 30 days." in sent["messages"][1][1]
    assert "Question: refund policy?" in sent["messages"][1][1]


def test_copies_of_one_chunk_do_not_use_up_the_result_slots(monkeypatch):
    same = LCDocument(page_content="Cost function title", metadata={"source": "a.pdf", "page": 39, "user_id": 1})
    real = LCDocument(page_content="Loss measures the error of a prediction.", metadata={"source": "a.pdf", "page": 41, "user_id": 1})

    class _Dupes:
        def as_retriever(self, **kwargs):
            return _FixedRetriever(docs=[same] * 6 + [real])

    monkeypatch.setattr(agent, "_vectorstore", _Dupes())
    chunks = agent.retrieve_chunks("loss function", user_id=1, k=3)
    assert [c["page"] for c in chunks] == [39, 41]


def test_plain_arithmetic_still_uses_the_agent(monkeypatch):
    monkeypatch.setattr(agent, "run_agent", lambda q, uid: "165")
    assert agent.run_agent_cited("what is 55 * 3", user_id=1) == ("165", [])
    assert agent._is_math("(12 + 8) / 4")
    assert not agent._is_math("what is 2024 revenue")
    assert not agent._is_math("what is machine learning?")


def test_the_model_is_shown_the_page(monkeypatch):
    seen = {}
    monkeypatch.setattr(agent, "_vectorstore", _VS())
    monkeypatch.setattr(agent, "get_llm", _fake_model)
    real = agent.create_retriever_tool

    def spy(*a, **k):
        tool = real(*a, **k)
        seen["out"] = tool.invoke({"query": "refund"})
        return tool

    monkeypatch.setattr(agent, "create_retriever_tool", spy)
    agent.build_agent(1)
    assert seen["out"] == "[1] policy.pdf, page 3\nRefunds take 30 days."


def test_chat_returns_citations_with_the_page(monkeypatch):
    chunks = [{"ref": 1, "source": "policy.pdf", "page": 3, "text": "Refunds take 30 days."}]
    monkeypatch.setattr(main, "run_agent_cited", lambda q, user_id: ("Refunds take 30 days. [1]", chunks))
    client = TestClient(main.app)
    token = client.post("/auth/register", json={"email": "cite1@example.com", "password": "a-real-password-123"}).json()["access_token"]

    body = client.post("/chat", json={"query": "refund policy?"}, headers={"Authorization": "Bearer " + token}).json()

    assert body["answer"] == "Refunds take 30 days. [1]"
    assert body["sources"] == ["policy.pdf p.3"]
    assert body["citations"] == [
        {"ref": 1, "source": "policy.pdf", "page": 3, "label": "policy.pdf p.3", "snippet": "Refunds take 30 days."}
    ]


def test_grouped_numbers_are_each_a_citation():
    chunks = [
        {"ref": 1, "source": "a.pdf", "page": 1, "text": "one"},
        {"ref": 2, "source": "a.pdf", "page": 2, "text": "two"},
        {"ref": 3, "source": "b.pdf", "page": None, "text": "three"},
    ]
    answer, citations, invalid = build_citations("Some claim [1, 2, 3].", chunks)
    assert answer == "Some claim [1][2][3]."
    assert [c["ref"] for c in citations] == [1, 2, 3]
    assert invalid == []


def test_grouped_number_that_was_never_retrieved_is_dropped():
    chunks = [{"ref": 1, "source": "a.pdf", "page": 1, "text": "one"}]
    answer, citations, invalid = build_citations("Claim [1, 7].", chunks)
    assert [c["ref"] for c in citations] == [1]
    assert invalid == [7]


def _tiny_pdf(page_texts):
    """A real, minimal PDF with one line of text per page."""
    objects = []
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(page_texts)))
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>")
    for i, text in enumerate(page_texts):
        content_id = 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents {content_id} 0 R "
            "/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>"
        )
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out = "%PDF-1.4\n"
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    return out.encode("latin-1")


def test_a_real_pdf_gives_the_right_text_on_the_right_page():
    pdf = _tiny_pdf(["Refunds take thirty days", "Shipping is free"])
    pages = text_processing.extract_pages(pdf, "policy.pdf")
    assert [n for n, _ in pages] == [1, 2]
    assert "Refunds" in pages[0][1] and "Shipping" in pages[1][1]


def test_identical_chunks_are_shown_once_and_the_limit_is_kept():
    same = LCDocument(page_content="Cost function intuition", metadata={"source": "a.pdf", "page": 39})
    other = LCDocument(page_content="Gradient descent steps", metadata={"source": "a.pdf", "page": 40})
    inner = _FixedRetriever(docs=[same, same, same, other])
    docs = agent.SourceTaggedRetriever(inner=inner, limit=2).invoke("loss")
    assert [d.page_content for d in docs] == ["Cost function intuition", "Gradient descent steps"]


def test_a_stuck_gemini_call_is_cancelled_quickly(monkeypatch):
    seen = {}
    monkeypatch.setattr(agent, "ChatGoogleGenerativeAI", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(agent.settings, "LLM_BACKEND", "gemini")
    agent.get_llm()
    assert seen["timeout"] == agent.settings.GEMINI_TIMEOUT_SECONDS <= 20
