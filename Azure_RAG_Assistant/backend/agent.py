from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools.retriever import create_retriever_tool
from config import settings, embeddings
from safe_math import safe_calculate

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

    vectorstore = PineconeVectorStore(index_name=settings.PINECONE_INDEX_NAME, embedding=embeddings)
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
# live Gemini/Pinecone credentials or network access.
_agent_graph = None


def get_agent():
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent()
    return _agent_graph


def run_agent(query: str) -> str:
    result = get_agent().invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content
