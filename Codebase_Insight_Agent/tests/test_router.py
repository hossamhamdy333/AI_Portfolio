import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
import portfolio


class FakeEmbedModel:
    """Every question scores 0 against every description, so only the
    keyword rule and the fallback can decide the result."""

    def get_text_embedding(self, text):
        if text in config.PROJECT_DESCRIPTIONS.values():
            return [1.0, 0.0]
        return [0.0, 1.0]


router = portfolio.ProjectRouter(FakeEmbedModel())


def test_skills_question_goes_to_the_overview():
    assert router.select("What skills does Hossam have?")[0] == config.OVERVIEW_PROJECT


def test_stack_question_goes_to_the_overview():
    assert router.select("what is your tech stack")[0] == config.OVERVIEW_PROJECT


def test_project_question_does_not_get_the_overview_added():
    picked = router.select("Which project uses LangGraph?")
    assert config.OVERVIEW_PROJECT not in picked


def test_something_is_always_picked():
    assert len(router.select("zzz")) == 1
