from qa_generation import generate_questions


def test_generate_questions_parses_valid_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeResponse:
        text = '{"qa_pairs": [{"question": "Q1", "answer": "A1"}]}'

    import qa_generation
    monkeypatch.setattr(qa_generation, "call_gemini", lambda *a, **kw: FakeResponse())

    questions, response = generate_questions(("model", "config"), "article text")
    assert questions == [{"question": "Q1", "answer": "A1"}]


def test_generate_questions_handles_malformed_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeResponse:
        text = "not valid json"

    import qa_generation
    monkeypatch.setattr(qa_generation, "call_gemini", lambda *a, **kw: FakeResponse())

    questions, response = generate_questions(("model", "config"), "article text")
    assert questions == []


def test_generate_questions_strips_markdown_fences(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeResponse:
        text = '```json\n{"qa_pairs": [{"question": "Q1", "answer": "A1"}]}\n```'

    import qa_generation
    monkeypatch.setattr(qa_generation, "call_gemini", lambda *a, **kw: FakeResponse())

    questions, response = generate_questions(("model", "config"), "article text")
    assert questions == [{"question": "Q1", "answer": "A1"}]
