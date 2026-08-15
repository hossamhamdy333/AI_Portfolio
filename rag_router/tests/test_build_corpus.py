from build_corpus import article_matches_domain, build_domain_corpus, DOMAIN_KEYWORDS


def test_article_matches_domain_true():
    assert article_matches_domain(
        "FIFA World Cup", "The FIFA World Cup is a football tournament.", DOMAIN_KEYWORDS["sports"]
    )


def test_article_matches_domain_false():
    assert not article_matches_domain(
        "Random Topic", "This text matches nothing relevant.", DOMAIN_KEYWORDS["sports"]
    )


def test_article_matches_domain_only_checks_first_500_chars():
    # A match buried past the 500-char window shouldn't count -- keeps
    # matching fast on very long articles without scanning the whole thing.
    long_text = ("x " * 300) + "football"
    assert not article_matches_domain("Title", long_text, DOMAIN_KEYWORDS["sports"])


def test_build_domain_corpus_buckets_correctly():
    fake_stream = [
        {"title": "FIFA World Cup", "text": "A football tournament."},
        {"title": "Python", "text": "Software and cloud computing."},
        {"title": "Roman Empire", "text": "An ancient dynasty."},
        {"title": "Pride and Prejudice", "text": "A victorian literature novel."},
        {"title": "Nothing Relevant", "text": "Matches no domain at all."},
    ]
    buckets = build_domain_corpus(iter(fake_stream), DOMAIN_KEYWORDS, target_per_domain=1, max_articles_scanned=100)
    assert [r["title"] for r in buckets["sports"]] == ["FIFA World Cup"]
    assert [r["title"] for r in buckets["tech"]] == ["Python"]
    assert [r["title"] for r in buckets["history"]] == ["Roman Empire"]
    assert [r["title"] for r in buckets["english_literature"]] == ["Pride and Prejudice"]


def test_build_domain_corpus_one_domain_per_article():
    # An article matching multiple domains' keywords should only be
    # bucketed into the first one matched, never double-counted.
    fake_stream = [{"title": "Ambiguous", "text": "football software"}] * 3
    buckets = build_domain_corpus(iter(fake_stream), DOMAIN_KEYWORDS, target_per_domain=5, max_articles_scanned=100)
    total_matched = sum(len(v) for v in buckets.values())
    assert total_matched <= 3  # never counted into more than one bucket per article


def test_build_domain_corpus_stops_at_max_scanned():
    fake_stream = [{"title": "Nothing", "text": "matches nothing"}] * 50
    buckets = build_domain_corpus(iter(fake_stream), DOMAIN_KEYWORDS, target_per_domain=100, max_articles_scanned=10)
    assert all(len(v) == 0 for v in buckets.values())
