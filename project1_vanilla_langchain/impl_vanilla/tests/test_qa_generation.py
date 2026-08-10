"""Tests for src/qa_generation.py -- prompt/parsing, no live API calls.
Cost/token math and retry logic are tested once, in shared/test_llm_client.py,
since qa_generation.py no longer defines its own copies of those functions.
"""

from src.qa_generation import generate_questions


def test_generate_questions_parses_valid_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")  # genai.Client() only checks a key is present at construction, no network call happens once call_gemini is mocked below

    class FakeResponse:
        text = '{"qa_pairs": [{"question": "q1", "answer": "a1"}]}'

    def fake_call_gemini(client, model_name, gen_config, prompt, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("src.qa_generation.call_gemini", fake_call_gemini)
    questions, response = generate_questions(("model", "config"), "some article text")
    assert questions == [{"question": "q1", "answer": "a1"}]


def test_generate_questions_handles_malformed_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeResponse:
        text = "not valid json"

    def fake_call_gemini(client, model_name, gen_config, prompt, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("src.qa_generation.call_gemini", fake_call_gemini)
    questions, response = generate_questions(("model", "config"), "some article text")
    assert questions == []
