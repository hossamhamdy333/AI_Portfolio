"""RouterQueryEngine over the 4 domain indexes.

Given a question, the router picks which domain index to query -- that
routing decision is itself scored (routing accuracy) alongside the usual
retrieval/generation metrics.
"""

from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

DOMAIN_DESCRIPTIONS = {
    "sports": "Useful for questions about sports, athletes, games, tournaments, and competitions.",
    "tech": "Useful for questions about technology, software, hardware, companies, and computing.",
    "history": "Useful for questions about historical events, figures, wars, and eras.",
    "english_literature": "Useful for questions about novels, poems, authors, literary movements, and English-language literature.",
}


def build_router(indexes, llm, top_k_retrieve=10):
    """indexes: dict of domain -> VectorStoreIndex (from ingest.py).

    Returns a RouterQueryEngine plus the ordered list of domain names, so a
    caller can map the tool selection back to a domain name for scoring.
    """
    domains = list(indexes.keys())
    tools = [
        QueryEngineTool.from_defaults(
            query_engine=indexes[domain].as_query_engine(llm=llm, similarity_top_k=top_k_retrieve),
            description=DOMAIN_DESCRIPTIONS.get(domain, f"Questions about {domain}."),
            name=domain,
        )
        for domain in domains
    ]

    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=tools,
        llm=llm,
    )
    return router, domains


def route_and_query(router, domains, question):
    """Run a question through the router.

    Returns (response_text, routed_domain, source_nodes) so the caller can
    check routed_domain against ground truth for routing accuracy, and pull
    source_nodes for citation verification the same way impl_vanilla does.
    """
    response = router.query(question)
    selector_result = response.metadata.get("selector_result") if response.metadata else None
    routed_domain = None
    if selector_result is not None and selector_result.selections:
        chosen_idx = selector_result.selections[0].index
        routed_domain = domains[chosen_idx]

    return str(response), routed_domain, response.source_nodes
