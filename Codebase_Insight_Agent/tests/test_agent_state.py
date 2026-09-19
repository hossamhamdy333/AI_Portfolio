"""
Each question must start from a clean agent state. The agent used to keep a
shared checkpoint, so feedback from one visitor's failed critique leaked
into the next visitor's first retrieval.
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import portfolio


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeEngine:
    def __init__(self, seen):
        self.seen = seen

    def query(self, query):
        self.seen.append(query)
        return "some context"


class FakeIndex:
    def __init__(self, seen):
        self.seen = seen

    def as_query_engine(self, similarity_top_k=5):
        return FakeEngine(self.seen)


class FakeRouter:
    def select(self, question):
        return ["rag_router"]


class FailingThenPassingLLM:
    """Fails the critique on the first question's attempts, passes afterwards."""

    critiques = 0

    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, prompt):
        if prompt.startswith("Is this answer"):
            FailingThenPassingLLM.critiques += 1
            if FailingThenPassingLLM.critiques <= 3:
                return FakeResponse("FAIL: missing detail")
            return FakeResponse("PASS")
        return FakeResponse("a draft answer")


def build_test_agent(seen):
    FailingThenPassingLLM.critiques = 0
    with patch("langchain_google_genai.ChatGoogleGenerativeAI", FailingThenPassingLLM):
        return portfolio.build_agent({"rag_router": FakeIndex(seen)}, FakeRouter())


def test_feedback_from_one_question_does_not_leak_into_the_next():
    seen = []
    agent = build_test_agent(seen)

    portfolio.ask(agent, "first visitor question")
    seen.clear()
    portfolio.ask(agent, "second visitor question")

    assert seen[0] == "second visitor question"


def test_llm_calls_counts_one_query_per_routed_project():
    seen = []
    agent = build_test_agent(seen)
    FailingThenPassingLLM.critiques = 10  # critique passes straight away

    result = portfolio.ask(agent, "any question")

    assert result["retries"] == 0
    assert result["llm_calls"] == 3  # one project query, one draft, one critique
