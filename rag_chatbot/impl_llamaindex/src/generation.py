"""Answer generation + citation checking for the router pipeline.

Unlike impl_vanilla (which asks the LLM to emit explicit [N] tags and parses
them out), the router pipeline checks citations by comparing the source
nodes LlamaIndex actually used against the question's ground-truth article
title. This is a deliberate difference, not a shortcut: LlamaIndex's
query engine doesn't naturally expose a numbered-sources prompt the way the
hand-rolled Gemini call does, and node-metadata matching is arguably a
*stricter* check (it verifies the retrieved chunk, not just a citation
number the model could hallucinate independent of what it retrieved).

route_and_query() in router.py does the actual work; this module just wraps
it with the same return shape (answer_text, citations) that impl_vanilla's
generate_answer() uses, so eval_routing.py and any future comparison code
can treat both implementations the same way.
"""

from router import route_and_query


def generate_answer(router, domains, question):
    """Thin wrapper around route_and_query for interface parity with impl_vanilla.

    Returns (answer_text, citations, routed_domain) where citations is a list
    of {"article_title": ...} dicts drawn from the source nodes actually used.
    """
    answer_text, routed_domain, source_nodes = route_and_query(router, domains, question)

    citations = [
        {"article_title": node.node.metadata.get("title"), "score": node.score}
        for node in (source_nodes or [])
    ]
    return answer_text, citations, routed_domain
