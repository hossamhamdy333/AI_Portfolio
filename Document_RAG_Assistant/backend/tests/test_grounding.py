"""
Grounding check: does each cited sentence match the chunk it cites?
It is a word-overlap check (no extra LLM call), so these tests also pin down
what it can and cannot catch.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")
os.environ.setdefault("QDRANT_URL", "http://test:6333")
os.environ.setdefault("QDRANT_API_KEY", "test")

from fastapi.testclient import TestClient

import main
from database import init_db
from grounding import check_grounding

init_db()

CHUNKS = [
    {"ref": 1, "source": "policy.pdf", "page": 3,
     "text": "Customers can request a refund within 30 days of purchase. "
             "Refunds are paid to the original payment method and take 5 to 7 business days."},
    {"ref": 2, "source": "hr.pdf", "page": 1,
     "text": "Employees receive 21 days of paid annual leave. Leave must be approved by the line manager."},
    {"ref": 3, "source": "ar.pdf", "page": 1, "text": "يمكن للعميل استرداد المبلغ خلال 30 يوما من الشراء"},
]


def test_a_sentence_that_matches_its_chunk_is_grounded():
    result = check_grounding("Refunds are paid to the original payment method. [1]", CHUNKS)
    assert result["status"] == "grounded"
    assert result["checks"][0]["ok"] and result["checks"][0]["support"] >= 0.8


def test_a_paraphrase_with_the_same_facts_passes():
    result = check_grounding("You can get a refund if you ask within 30 days of buying. [1]", CHUNKS)
    assert result["status"] == "grounded"


def test_a_wrong_number_is_caught():
    result = check_grounding("Refunds are possible within 60 days of purchase. [1]", CHUNKS)
    assert result["status"] == "weak"
    assert result["checks"][0]["missing_numbers"] == ["60"]


def test_a_sentence_the_chunk_never_mentions_is_caught():
    result = check_grounding("The company offers free parking for all staff. [1]", CHUNKS)
    assert result["status"] == "weak" and result["checks"][0]["support"] == 0.0


def test_a_real_fact_cited_to_the_wrong_chunk_is_caught():
    result = check_grounding("Annual leave is 21 days. [1]", CHUNKS)
    assert result["status"] == "weak"
    assert result["checks"][0]["missing_numbers"] == ["21"]


def test_each_sentence_is_checked_against_its_own_chunk():
    result = check_grounding("Refunds go back to the original payment method. [1] Annual leave is 21 days. [2]", CHUNKS)
    assert result["status"] == "grounded"
    assert [c["refs"] for c in result["checks"]] == [[1], [2]]


def test_one_bad_sentence_makes_the_whole_answer_weak():
    result = check_grounding("Annual leave is 21 days. [2] Refunds take 90 days. [1]", CHUNKS)
    assert result["status"] == "weak"
    assert [c["ok"] for c in result["checks"]] == [True, False]


def test_a_sentence_citing_two_chunks_can_use_both():
    result = check_grounding("Leave needs manager approval and refunds take 5 to 7 days. [1][2]", CHUNKS)
    assert result["status"] == "grounded"
    assert result["checks"][0]["refs"] == [1, 2]


def test_a_marker_in_the_middle_of_a_sentence_splits_the_claims():
    result = check_grounding("Refunds take 5 to 7 business days [1], and leave is 21 days [2].", CHUNKS)
    assert result["status"] == "grounded" and len(result["checks"]) == 2


def test_non_english_text_is_checked_too():
    assert check_grounding("يمكن استرداد المبلغ خلال 30 يوما [3]", CHUNKS)["status"] == "grounded"
    assert check_grounding("يمكن استرداد المبلغ خلال 90 يوما [3]", CHUNKS)["status"] == "weak"


def test_an_answer_with_no_citations_is_unchecked_not_failed():
    assert check_grounding("55 * 3 = 165", CHUNKS) == {"status": "unchecked", "score": None, "checks": [], "invalid_refs": []}


def test_a_made_up_chunk_number_makes_it_weak():
    result = check_grounding("Refunds take 5 to 7 business days. [1]", CHUNKS, invalid_refs=[9])
    assert result["status"] == "weak" and result["invalid_refs"] == [9]


def test_chat_reports_a_weak_answer(monkeypatch):
    chunks = [{"ref": 1, "source": "policy.pdf", "page": 3, "text": "Refunds are possible within 30 days of purchase."}]
    monkeypatch.setattr(main, "run_agent_cited", lambda q, user_id: ("Refunds are possible within 60 days. [1]", chunks))
    client = TestClient(main.app)
    token = client.post("/auth/register", json={"email": "ground1@example.com", "password": "a-real-password-123"}).json()["access_token"]

    body = client.post("/chat", json={"query": "refund window?"}, headers={"Authorization": "Bearer " + token}).json()

    assert body["grounding"]["status"] == "weak"
    assert body["grounding"]["checks"][0]["missing_numbers"] == ["60"]
    assert body["citations"][0]["page"] == 3


def test_chat_reports_a_grounded_answer(monkeypatch):
    chunks = [{"ref": 1, "source": "policy.pdf", "page": 3, "text": "Refunds are possible within 30 days of purchase."}]
    monkeypatch.setattr(main, "run_agent_cited", lambda q, user_id: ("Refunds are possible within 30 days. [1]", chunks))
    client = TestClient(main.app)
    token = client.post("/auth/register", json={"email": "ground2@example.com", "password": "a-real-password-123"}).json()["access_token"]

    body = client.post("/chat", json={"query": "refund window?"}, headers={"Authorization": "Bearer " + token}).json()

    assert body["grounding"]["status"] == "grounded"
