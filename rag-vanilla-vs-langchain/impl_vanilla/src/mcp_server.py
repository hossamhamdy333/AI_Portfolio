"""MCP server exposing the RAG chatbot as a tool for any MCP-compatible
client (Claude Desktop, etc.), instead of only being reachable via
api.py's HTTP endpoint.

This wraps the exact same retrieval.py + generation.py functions api.py
uses -- same Qdrant Cloud collection, same reranker, same prompt -- just
behind the Model Context Protocol instead of (or alongside) REST. Because
it connects to Qdrant Cloud rather than a local embedded instance, this
can run from any machine with the right env vars set, not just the Colab
session that built the collection.

Run: python -m src.mcp_server
Requires QDRANT_API_KEY and GEMINI_API_KEY in the environment.
"""

import logging
import os

import yaml
from google import genai
from mcp.server import MCPServer
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.generation import generate_answer
from src.retrieval import rerank_chunks, search_chunks

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config.yaml")
with open(_CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# mcp.server.fastmcp.FastMCP (the older SDK's convenience class) was
# renamed/relocated to mcp.server.MCPServer as of mcp 2.0.0 -- verified
# against the actual currently-published package, not assumed from
# training data. Same .tool() decorator ergonomics, just a different name
# and import path.
mcp = MCPServer("documind-arabic-news")

# Loaded lazily on first tool call, not at import time -- consistent with
# how shared/llm_client.py's get_client() and qa_generation.py's client
# creation were fixed earlier in this repo (importing this module for
# testing, or MCP's own tool-discovery step, shouldn't require a live
# API key or a running Qdrant Cloud connection).
_state = {}


def _get_state():
    if not _state:
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        if not qdrant_api_key:
            raise RuntimeError(
                "QDRANT_API_KEY is not set. Get one from your free cluster at "
                "https://cloud.qdrant.io."
            )
        _state["embedder"] = SentenceTransformer(CONFIG["retrieval"]["model_name"])
        _state["reranker"] = CrossEncoder(CONFIG["reranker"]["model_name"])
        _state["qdrant_client"] = QdrantClient(url=CONFIG["qdrant"]["url"], api_key=qdrant_api_key)
        _state["gemini_client"] = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _state


@mcp.tool()
def ask_arabic_news(question: str) -> str:
    """Answer a question about the indexed Arabic news corpus (XLSum),
    citing the source article for every claim.

    Args:
        question: A question in Arabic (or English -- the retriever is
            multilingual, but the corpus and answers are Arabic).
    """
    state = _get_state()

    candidates = search_chunks(
        question, state["qdrant_client"], CONFIG["qdrant"]["collection_name"],
        state["embedder"], top_k=CONFIG["rag"]["top_k_retrieve"],
    )
    if not candidates:
        return "لا توجد معلومات كافية للإجابة على هذا السؤال."

    reranked = rerank_chunks(question, candidates, state["reranker"], top_k=CONFIG["rag"]["top_k_rerank"])

    answer_text, citations, _ = generate_answer(
        state["gemini_client"], question, reranked,
        model_name=CONFIG["rag"]["llm_model"],
        temperature=CONFIG["rag"]["temperature"],
        max_output_tokens=CONFIG["rag"]["max_output_tokens"],
    )
    if citations:
        sources = ", ".join(sorted({c["article_id"] for c in citations}))
        answer_text += f"\n\n[sources: {sources}]"
    return answer_text


if __name__ == "__main__":
    # stdio transport -- what Claude Desktop and most MCP clients expect
    # for a locally-run server added to their config. Switch to
    # mcp.run(transport="sse") if this needs to be reachable over HTTP
    # instead of spawned as a subprocess.
    mcp.run(transport="stdio")
