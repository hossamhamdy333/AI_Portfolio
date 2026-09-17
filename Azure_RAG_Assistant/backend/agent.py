from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue, PayloadSchemaType
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


def get_llm():
    """Picks the LLM backend. Gemini by default; set LLM_BACKEND=local to
    point at Ollama instead (`ollama serve`, then `ollama pull llama3.1`) -
    see the load test in scripts/load_test.py for comparing the two on
    latency and cost."""
    if settings.LLM_BACKEND == "local":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=settings.LOCAL_LLM_BASE_URL,
            api_key="not-needed",  # Ollama doesn't check this
            model=settings.LOCAL_LLM_MODEL,
            temperature=0.1,
        )
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.1,
    )


# The Qdrant client is stateless and thread-safe, so one shared instance
# is reused across every request instead of reconnecting each time. The
# retriever itself is NOT shared, though - see build_agent below - because
# it's what carries the per-user filter, and that has to be different for
# every user.
_qdrant_client = None


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        if not _qdrant_client.collection_exists(settings.QDRANT_COLLECTION_NAME):
            # The collection is normally created on first document upload
            # (text_processing.py). If someone chats before uploading
            # anything, it won't exist yet - create an empty one here so
            # the agent doesn't crash on startup with no documents indexed.
            _qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIMENSIONS, distance=Distance.COSINE),
            )
            # Every search in build_agent() below filters on this field to
            # keep users' documents isolated from each other. Qdrant
            # refuses to filter on a field with no index, so without this
            # line every chat request fails with "Index required but not
            # found for metadata.user_id" the moment a second user (or the
            # first query on a fresh collection) shows up.
            _qdrant_client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                field_name="metadata.user_id",
                field_schema=PayloadSchemaType.INTEGER,
            )
    return _qdrant_client


def build_agent(user_id: int, top_k=3):
    """A fresh agent per call, scoped to one user's documents. Rebuilding
    this per request is cheap (it's just wiring together already-built
    pieces, not loading anything) - the real cost is the LLM call itself,
    which this doesn't add to.

    top_k is exposed as a parameter (not just an internal constant) so
    scripts/regression_demo.py can deliberately degrade retrieval (top_k=1)
    and show the difference in the observability trace.
    """
    llm = get_llm()
    vectorstore = QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.QDRANT_COLLECTION_NAME,
        embedding=embeddings,
    )

    # This filter is the actual security boundary - it's checked by
    # Qdrant itself on every search, not just something the UI happens to
    # respect. See tests/test_agent_isolation.py for proof this can't be
    # bypassed by asking about "all documents" or similar.
    user_filter = Filter(must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))])
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k, "filter": user_filter})

    retriever_tool = create_retriever_tool(
        retriever,
        "company_knowledge_base",
        "Search this user's previously uploaded documents for relevant context.",
    )

    return create_agent(
        model=llm,
        tools=[retriever_tool, calculator],
        system_prompt=SYSTEM_PROMPT,
    )


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


def run_agent(query: str, user_id: int) -> str:
    agent = build_agent(user_id)
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return _extract_text(result["messages"][-1].content)
