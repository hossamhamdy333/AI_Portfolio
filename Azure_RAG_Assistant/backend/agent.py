import re
import threading

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue, PayloadSchemaType
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools.retriever import create_retriever_tool
from config import settings, embeddings, logger
from safe_math import safe_calculate

# Must match the embedding model's output dimensionality
# (sentence-transformers/all-MiniLM-L6-v2 -> 384).
EMBEDDING_DIMENSIONS = 384

SYSTEM_PROMPT = (
    "You are Azure RAG Assistant, a helpful enterprise assistant. "
    "Always check the company_knowledge_base tool first for questions that "
    "might be answered by uploaded documents. Use the calculator tool for "
    "any arithmetic. Answer concisely. When your answer used the knowledge "
    "base, end with one final line in exactly this form: "
    "Source: <document name>  (separate several names with commas). Use the "
    "document names shown in the knowledge base results, and put nothing "
    "after that line. If you did not use the knowledge base, do not write a "
    "Source line."
)

# How each retrieved chunk is shown to the model. Including the document's
# real filename lets it cite "Hossam_Hamdy_CV.pdf" instead of paraphrasing
# ("Hossam's Resume").
SOURCE_DOCUMENT_PROMPT = PromptTemplate.from_template("[Document: {source}]\n{page_content}")


class SourceTaggedRetriever(BaseRetriever):
    """Wraps a retriever so every chunk it returns has a 'source' name.

    The prompt above requires one, and a chunk without it (an old upload,
    say) would otherwise make the whole search fail with a missing-variable
    error. A real filename is never overwritten.
    """

    inner: BaseRetriever

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        docs = self.inner.invoke(query, config={"callbacks": run_manager.get_child()})
        for doc in docs:
            doc.metadata.setdefault("source", "unknown document")
        return docs


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
        timeout=60,      # a stuck call fails after a minute instead of hanging the request forever
        max_retries=3,   # the default of 6 can keep one request waiting for minutes under rate limiting
    )


# The Qdrant client is stateless and thread-safe, so one shared instance
# is reused across every request instead of reconnecting each time. The
# retriever itself is NOT shared, though - see build_agent below - because
# it's what carries the per-user filter, and that has to be different for
# every user.
_qdrant_client = None
_vectorstore = None
# Chat and upload run in worker threads (several at once), so first-time
# setup must not run twice in parallel - two threads both creating the
# collection would make one of them fail with "already exists".
_init_lock = threading.Lock()

# Fields every per-user search / per-document delete filters on. Qdrant
# refuses to filter on a field with no index ("Index required but not found
# for metadata.user_id"), so both need one.
_INDEXED_FIELDS = ("metadata.user_id", "metadata.document_id")


def _ensure_payload_indexes(client) -> None:
    """Creating an index that already exists is a harmless no-op, so this is
    safe to run on every start. It has to run even when the collection
    already exists: the first document upload creates the collection
    itself (text_processing.py) WITHOUT these indexes, and the next chat
    would then fail."""
    for field in _INDEXED_FIELDS:
        try:
            client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.INTEGER,
            )
        except Exception as exc:
            logger.warning("Could not ensure the Qdrant index on %s: %s", field, str(exc)[:160])


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    with _init_lock:
        if _qdrant_client is None:
            client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            if not client.collection_exists(settings.QDRANT_COLLECTION_NAME):
                # The collection is normally created on first document upload
                # (text_processing.py). If someone chats before uploading
                # anything, it won't exist yet - create an empty one here so
                # the agent doesn't crash on startup with no documents indexed.
                client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    vectors_config=VectorParams(size=EMBEDDING_DIMENSIONS, distance=Distance.COSINE),
                )
            _ensure_payload_indexes(client)
            _qdrant_client = client
    return _qdrant_client


def _get_vectorstore():
    """One shared vector store. Building it makes a network round trip to
    Qdrant (it checks the collection's configuration), which used to
    happen again on every single chat message. It holds no per-user state -
    the per-user filter is applied on the retriever in build_agent() - so it
    is safe to share."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    with _init_lock:
        if _vectorstore is None:
            _vectorstore = QdrantVectorStore(
                client=get_qdrant_client(),
                collection_name=settings.QDRANT_COLLECTION_NAME,
                embedding=embeddings,
            )
    return _vectorstore


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
    vectorstore = _get_vectorstore()

    # This filter is the actual security boundary - it's checked by
    # Qdrant itself on every search, not just something the UI happens to
    # respect. See tests/test_agent_isolation.py for proof this can't be
    # bypassed by asking about "all documents" or similar.
    user_filter = Filter(must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))])
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k, "filter": user_filter})

    retriever_tool = create_retriever_tool(
        SourceTaggedRetriever(inner=retriever),
        "company_knowledge_base",
        "Search this user's previously uploaded documents for relevant context.",
        document_prompt=SOURCE_DOCUMENT_PROMPT,
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


# ---------------------------------------------------------------- sources

_SOURCE_LINE = re.compile(r"^\W*(?:sources?|source documents?)\s*:\s*(.+)$", re.IGNORECASE)
_NO_SOURCE = {"none", "n/a", "na", "unknown", "-", "no source", "no sources"}
MAX_SOURCES = 5


def split_sources(text: str) -> tuple[str, list[str]]:
    """Separates a trailing "Source: ..." line from an answer.

    The model is asked to end a knowledge-base answer with one line such as
    "Source: report.pdf, notes.txt" (a few variants - "(Source: ...)",
    "**Sources:** ...", "Sources: a and b" - are understood too). It comes
    back separately so the page can show it as "From:" badges instead of
    a sentence inside the answer. Only the LAST line is considered, so a
    sentence that merely contains the word "source" is never touched, and if
    there is no such line the answer is returned unchanged.
    """
    lines = text.rstrip().split("\n")
    if not lines:
        return text, []

    last = lines[-1].strip()
    match = _SOURCE_LINE.match(last)
    if not match:
        return text, []

    rest = match.group(1).strip().strip("*_)").strip().rstrip(".").strip()
    body = "\n".join(lines[:-1]).rstrip()
    if not body:  # the whole reply was just a citation: keep it as the answer
        return text, []

    names = []
    for name in re.split(r",|;|\s+and\s+|\s+&\s+", rest):
        name = name.strip().strip("*_()[]\"' ").strip()
        if name and name.lower() not in _NO_SOURCE and name not in names:
            names.append(name[:80])
    return body, names[:MAX_SOURCES]
