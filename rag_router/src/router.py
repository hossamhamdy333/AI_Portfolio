"""RouterQueryEngine over the 4 domain indexes.

Given a question, the router picks which domain index to query -- that
routing decision is itself scored (routing_accuracy) alongside the usual
retrieval/generation metrics.

Two selector options:
  - build_router(): LLMSingleSelector -- asks the LLM to pick a domain.
  - build_embedding_router(): EmbeddingSingleSelector -- picks by comparing
    the question's embedding to each domain description's embedding. No LLM
    call for routing at all, so it's faster/cheaper per question.

Both selectors are actually evaluated in notebooks/04_evaluation.ipynb, and
COMPARISON.md reports both numbers -- an earlier version of this docstring
claimed a specific routing accuracy for the LLM selector while the eval
notebook that produced that number had actually called
build_embedding_router() instead. Whichever selector a given number in
COMPARISON.md is attributed to, that's the one that was actually run.

Both selectors can hit the same underlying failure mode: LlamaIndex's
selector-result parsing (`Answer`/`SingleSelection`, via dataclasses_json)
occasionally decodes a response with `choice=None`, which crashes a few
frames later with `TypeError: unsupported operand type(s) for -: 'NoneType'
and 'int'` when LlamaIndex converts that choice to a 0-indexed domain --
this reproduces under EmbeddingSingleSelector too, not just the LLM one, so
"no LLM call for routing" does not mean "immune to this parsing bug." It's
deterministic for a given question (same input reliably fails the same
way), not a transient rate-limit-style error -- see eval_routing.py's
_route_with_retry(), which now short-circuits on this specific error
instead of burning 5 identical retries that can never succeed.
"""

from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import EmbeddingSingleSelector, LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

DOMAIN_DESCRIPTIONS = {
    "sports": (
        "Sports, athletes, teams, games, matches, tournaments, championships, "
        "leagues, coaches, stadiums, and athletic competitions of any kind "
        "(football, basketball, tennis, cricket, olympics, etc.)."
    ),
    "tech": (
        "Technology, computer science, software, hardware, the internet, "
        "programming, artificial intelligence, semiconductors, smartphones, "
        "tech companies, and computing systems. Not sports technology or "
        "historical inventions -- only modern computing/tech topics."
    ),
    "history": (
        "Historical events, wars, empires, revolutions, dynasties, ancient "
        "and medieval civilizations, treaties, and figures from past eras. "
        "Not current events or modern technology history."
    ),
    "english_literature": (
        "Novels, poems, plays, authors, poets, playwrights, literary "
        "movements, and works of English-language fiction or poetry. "
        "Not historical events themselves, only literary works about them."
    ),
}


def _build_tools(indexes, llm, top_k_retrieve):
    domains = list(indexes.keys())
    tools = [
        QueryEngineTool.from_defaults(
            query_engine=indexes[domain].as_query_engine(llm=llm, similarity_top_k=top_k_retrieve),
            description=DOMAIN_DESCRIPTIONS.get(domain, f"Questions about {domain}."),
            name=domain,
        )
        for domain in domains
    ]
    return tools, domains


def build_router(indexes, llm, top_k_retrieve=10):
    """indexes: dict of domain -> VectorStoreIndex (from ingest.py).

    Returns a RouterQueryEngine plus the ordered list of domain names, so a
    caller can map the tool selection back to a domain name for scoring.
    """
    tools, domains = _build_tools(indexes, llm, top_k_retrieve)

    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=tools,
        llm=llm,
    )
    return router, domains


def build_embedding_router(indexes, llm, embed_model, top_k_retrieve=10):
    """Same tools/domains as build_router(), but routes by embedding
    similarity instead of an LLM call -- doesn't spend an API call on
    routing. See this module's docstring: this does NOT avoid the
    malformed-selector-response failure mode, only the LLM-call cost.
    """
    tools, domains = _build_tools(indexes, llm, top_k_retrieve)

    router = RouterQueryEngine(
        selector=EmbeddingSingleSelector.from_defaults(embed_model=embed_model),
        query_engine_tools=tools,
        llm=llm,
    )
    return router, domains


def route_and_query(router, domains, question):
    """Run a question through the router.

    Returns (response_text, routed_domain, source_nodes) so the caller can
    check routed_domain against ground truth for routing accuracy, and pull
    source_nodes for citation verification.
    """
    response = router.query(question)
    selector_result = response.metadata.get("selector_result") if response.metadata else None
    routed_domain = None
    if selector_result is not None and selector_result.selections:
        chosen_idx = selector_result.selections[0].index
        routed_domain = domains[chosen_idx]

    return str(response), routed_domain, response.source_nodes
