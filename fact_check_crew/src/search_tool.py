"""Search tool for the Researcher agent.

Searches across all 4 of ../rag_router's existing Wikipedia domain
collections directly via qdrant-client -- no LlamaIndex query engine here,
since the Researcher just needs "give me relevant passages," not a routed
query engine or a synthesized answer. Reusing rag_router's LlamaIndex
machinery for that would be pulling in a whole retrieval framework to do
something a plain vector search already does. Embeddings come from
sentence-transformers directly (the same model rag_router used to build
these collections, just without LlamaIndex's wrapper around it -- one
less dependency for a project that isn't using the rest of LlamaIndex).

Payload shape: LlamaIndex's QdrantVectorStore doesn't store a flat "text"
field -- the actual passage text is nested inside a JSON-stringified
`_node_content` field (LlamaIndex's internal node serialization).
`title`/`domain` ARE flat top-level keys (they came from the Document's
metadata dict), but `text` isn't -- confirmed by actually building a real
collection with rag_router's ingest.py and inspecting the raw payload,
not assumed from documentation.
"""

import json

from crewai.tools import tool


def _extract_text(payload):
    node_content = payload.get("_node_content")
    if not node_content:
        return ""
    try:
        return json.loads(node_content).get("text", "")
    except (json.JSONDecodeError, AttributeError):
        return ""


def search_domains(query, qdrant_client, embed_model, domains, collection_prefix, top_k=5):
    """Embed the query once, search every domain collection, return the
    top_k passages overall by score. Plain function (not the @tool-wrapped
    one below) so it's easy to unit test without CrewAI in the loop.
    """
    query_vector = embed_model.encode(query).tolist()

    all_hits = []
    for domain in domains:
        collection_name = f"{collection_prefix}_{domain}"
        if not qdrant_client.collection_exists(collection_name):
            continue
        results = qdrant_client.query_points(collection_name=collection_name, query=query_vector, limit=top_k).points
        for r in results:
            all_hits.append({
                "domain": domain,
                "title": r.payload.get("title", "unknown"),
                "text": _extract_text(r.payload),
                "score": r.score,
            })

    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return all_hits[:top_k]


def format_hits_for_agent(hits):
    """Plain-text block the Researcher agent's tool call returns -- CrewAI
    tools return strings, not structured objects, since that's what an LLM
    reads.
    """
    if not hits:
        return "No relevant passages found."
    lines = []
    for i, h in enumerate(hits, start=1):
        lines.append(f"[{i}] ({h['domain']}, \"{h['title']}\"): {h['text'][:500]}")
    return "\n\n".join(lines)


def build_search_tool(qdrant_client, embed_model, domains, collection_prefix, top_k=5):
    """Returns a CrewAI tool closed over the already-connected client/model,
    so the agent only ever sees `search_wikipedia(query)` -- no
    connection details leak into what the LLM can see or call.
    """

    @tool("search_wikipedia")
    def search_wikipedia(query: str) -> str:
        """Search indexed Wikipedia articles across sports, tech, history,
        and English literature. Returns the most relevant passages found,
        each tagged with its domain and title."""
        hits = search_domains(query, qdrant_client, embed_model, domains, collection_prefix, top_k)
        return format_hits_for_agent(hits)

    return search_wikipedia
