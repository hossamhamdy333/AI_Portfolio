"""Citation parsing + verification for the LangChain pipeline.

Renamed from generation.py -- this module was named the same as
impl_vanilla's generation.py despite doing something different (chain.py
does the actual LLM generation via LCEL; this only re-checks the model's
[N] tags against what was actually retrieved). The shared name across two
implementations that mean different things was more confusing than useful
once they sat side by side in one repo.

Same approach as impl_vanilla: re-check citations against what was
actually retrieved, rather than trusting the generated citations at face
value.
"""

import re


def extract_cited_indices(answer_text):
    """Pull all [N] tags out of the answer, 1-indexed as written in the prompt."""
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer_text)})


def verify_citations(answer_text, reranked_docs, correct_article_id):
    """Checks whether the article actually matching the question's ground
    truth is among the docs the model cited -- same correctness definition
    impl_vanilla uses, so citation_accuracy is comparable across COMPARISON.md.
    """
    cited_indices = extract_cited_indices(answer_text)
    cited_article_ids = {
        reranked_docs[i - 1].metadata.get("article_id")
        for i in cited_indices
        if 0 < i <= len(reranked_docs)
    }
    return {
        "cited_indices": cited_indices,
        "cited_article_ids": list(cited_article_ids),
        "correct_cited": correct_article_id in cited_article_ids,
    }
