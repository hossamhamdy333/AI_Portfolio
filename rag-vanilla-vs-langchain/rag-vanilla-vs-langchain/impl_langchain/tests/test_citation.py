"""Tests for src/citation.py -- no live API calls, just tag parsing/matching."""

from citation import extract_cited_indices, verify_citations


class FakeDoc:
    def __init__(self, article_id):
        self.metadata = {"article_id": article_id}


def test_extract_cited_indices_finds_all_tags():
    text = "الجواب هو كذا [1] وأيضا هذا [3] وهذا [1]"
    assert extract_cited_indices(text) == [1, 3]


def test_extract_cited_indices_no_tags():
    assert extract_cited_indices("لا يوجد استشهادات هنا") == []


def test_verify_citations_correct_article_cited():
    docs = [FakeDoc("art_a"), FakeDoc("art_b")]
    result = verify_citations("الجواب [1]", docs, correct_article_id="art_a")
    assert result["correct_cited"] is True
    assert result["cited_article_ids"] == ["art_a"]


def test_verify_citations_wrong_article_cited():
    docs = [FakeDoc("art_a"), FakeDoc("art_b")]
    result = verify_citations("الجواب [2]", docs, correct_article_id="art_a")
    assert result["correct_cited"] is False


def test_verify_citations_out_of_range_index_ignored():
    docs = [FakeDoc("art_a")]
    result = verify_citations("الجواب [5]", docs, correct_article_id="art_a")
    assert result["cited_article_ids"] == []
    assert result["correct_cited"] is False
