from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_core.tools.retriever import create_retriever_tool
from config import settings, embeddings
from safe_math import safe_calculate

# Must match the embedding model's output dimensionality
# (sentence-transformers/all-MiniLM-L6-v2 -> 384).
EMBEDDING_DIMENSIONS = 384

SYSTEM_PROMPT = (
    "You are Azure RAG Assistant, a helpful enterprise assistant. "
    "Always check the company_knowledge_base tool first for questions that "
    "might be answered by uploaded documents. Use the calculator tool for "
    "any arithmetic. Answer concisely and cite which document a fact came "
    "from when you used the knowledge base."
)


@tool
def calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression, e.g. '55 * 3' or '(12 + 8) / 4'."""
    return safe_calculate(expression)


def build_agent():
    llm = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.1,
    )

    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    # The collection is normally created on first document upload
    # (text_processing.py). If someone chats before uploading anything, it
    # won't exist yet - create an empty one here so the agent doesn't crash
    # on startup with no documents indexed.
    if not client.collection_exists(settings.QDRANT_COLLECTION_NAME):
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSIONS, distance=Distance.COSINE),
        )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retriever_tool = create_retriever_tool(
        retriever,
        "company_knowledge_base",
        "Search previously uploaded company documents for relevant context.",
    )

    return create_agent(
        model=llm,
        tools=[retriever_tool, calculator],
        system_prompt=SYSTEM_PROMPT,
    )


# Built lazily on first use (not at import time) so importing this module -
# e.g. from a test that only checks the /health endpoint - never requires
# live Gemini/Qdrant credentials or network access.
_agent_graph = None


def get_agent():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent()
    return _agent_graph


def _extract_text(content) -> str:
    """
    Newer Gemini models can return message content as either a plain string
    or a list of structured content parts (e.g. [{"type": "text", "text":
    "..."}]) instead of always a string. Normalize either shape into plain
    text so the frontend always receives a string, not an object it would
    otherwise render as "[object Object]".
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts) if parts else str(content)
    return str(content)


def run_agent(query: str) -> str:
    result = get_agent().invoke({"messages": [{"role": "user", "content": query}]})
    return _extract_text(result["messages"][-1].content)
