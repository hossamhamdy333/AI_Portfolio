# The actual logic, in one plain file so every notebook can just
# `import portfolio` instead of repeating this code five times.

from pathlib import Path
from typing import TypedDict

import requests

import config

DATA_DIR = Path(__file__).parent / "data"

_llama_index_configured = False


def configure_llama_index():
    """Sets llama_index's global embed_model/llm.

    embed_model is a local, free, no-API-key sentence-transformers model
    (config.EMBEDDING_MODEL) - runs on CPU, no rate limit, no quota,
    same model already used elsewhere in this portfolio
    (rag_router/fact_check_crew's shared corpus, Azure_RAG_Assistant,
    customer_support_copilot). llm stays on Gemini (config.LLM_MODEL) -
    only embedding was ever hitting the free-tier quota wall during
    indexing, chat generation is a separate, much lower-volume call.

    Without setting embed_model explicitly, VectorStoreIndex.from_documents()
    and as_query_engine() both fall back to llama_index's built-in default,
    which tries to resolve OpenAI's embedding class - raising
    `ImportError: llama-index-embeddings-openai package not found`
    immediately, since this project never installs OpenAI's integration.
    Nothing else in this file calls this for you implicitly;
    build_index/load_index/as_query_engine callers all call it themselves
    first (it's idempotent - safe to call every time, only does real work
    once per process).
    """
    global _llama_index_configured
    if _llama_index_configured:
        return

    from llama_index.core import Settings
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
    Settings.llm = GoogleGenAI(model=config.LLM_MODEL)
    _llama_index_configured = True


def _extract_text(response):
    """Normalizes an LLM response's .content into plain text.

    Some Gemini models return .content as a plain string; others return
    a list of structured parts (e.g. [{"type": "text", "text": "..."}]).
    Every caller in this file needs plain text, so this is the one place
    that knows how to handle both shapes.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def get_readme(project_name):
    """Get a project's real content from GitHub: its main README, plus any
    real supplementary docs listed in config.PROJECT_FILES (a COMPARISON.md,
    a reports/ writeup, a sub-implementation's own README), concatenated
    into one document with a clear header marking where each supplementary
    piece starts. Two projects (the graduation project's own two repos)
    live outside AI_Portfolio entirely, via config.PROJECT_REPO_OVERRIDES -
    everything else defaults to a subfolder of this repo.

    Falls back to a saved copy in data/ if GitHub can't be reached, or if
    any file in the list fails partway through - better to fall back to
    the last-known-good combined copy than index a half-fetched document
    with some supplementary sections silently missing."""
    repo = config.PROJECT_REPO_OVERRIDES.get(project_name)
    branch = config.PROJECT_BRANCH_OVERRIDES.get(project_name, config.GITHUB_BRANCH)
    files = config.PROJECT_FILES.get(project_name, ["README.md"])

    if repo:
        # A standalone repo of its own - files live at the repo root.
        base_url = f"https://raw.githubusercontent.com/{config.GITHUB_ORG}/{repo}/{branch}"
    else:
        # A subfolder of the main portfolio repo.
        base_url = f"https://raw.githubusercontent.com/{config.GITHUB_ORG}/{config.GITHUB_REPO}/{branch}/{project_name}"

    try:
        parts = []
        for i, file_path in enumerate(files):
            response = requests.get(f"{base_url}/{file_path}", timeout=10)
            response.raise_for_status()
            if i == 0:
                parts.append(response.text)
            else:
                parts.append(f"\n\n---\n\n# Supplementary document: `{file_path}`\n\n{response.text}")
        return "".join(parts)
    except requests.RequestException:
        path = DATA_DIR / f"{project_name}_readme_fixture.md"
        print(f"Couldn't reach GitHub for {project_name}, using the saved copy instead")
        return path.read_text()


def split_into_chunks(text, chunk_size=512, overlap=64):
    """Split text into chunks, keeping whole paragraphs together
    where possible, and only breaking a paragraph up if it's too long."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
            continue
        start = 0
        while start < len(para):
            chunks.append(para[start:start + chunk_size])
            start += chunk_size - overlap
    return chunks


def get_qdrant_client():
    """One real client, shared across every call in this process. Points
    at a persistent Qdrant Cloud instance if QDRANT_URL is set (same
    pattern Azure_RAG_Assistant already uses) - this is what makes an
    index built by running 01_indexing.ipynb still be there the next
    time mcp_server.py or web_app.py starts, instead of every process
    restart silently re-embedding all 19 projects from scratch.

    Falls back to a local in-memory client if QDRANT_URL isn't set, for
    zero-setup quick testing - but that mode has NO persistence at all:
    every process restart rebuilds everything, and the whole point of
    01_indexing.ipynb (provisioning a persistent index the live services
    just load) doesn't apply until a real QDRANT_URL is configured.
    """
    from qdrant_client import QdrantClient

    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    return QdrantClient(":memory:")


def collection_name(project_name):
    return f"portfolio_{project_name}"


def index_exists(project_name, client=None):
    client = client or get_qdrant_client()
    return client.collection_exists(collection_name(project_name))


def build_index(project_name, client=None):
    """Build a searchable index over one project's README - fetches the
    README, embeds every chunk, writes it to Qdrant. This is the
    expensive path (real API calls, real latency) - see load_index()
    below for the cheap path that reads what this already wrote.

    Drops the collection first if it already exists - re-embedding into
    an existing collection doesn't replace the old points, it silently
    adds new ones alongside them (verified this directly: rebuilding
    with different content took a collection from 1 point to 2, not a
    clean 1-to-1 replacement). Without the drop, a force=True "refresh
    after editing a README" would leave stale old chunks still
    searchable next to the new ones, not actually refresh anything."""
    configure_llama_index()
    from llama_index.core import Document, VectorStoreIndex, StorageContext
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    client = client or get_qdrant_client()
    name = collection_name(project_name)
    if client.collection_exists(name):
        client.delete_collection(name)

    text = get_readme(project_name)
    chunks = split_into_chunks(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    docs = [Document(text=chunk, metadata={"project": project_name}) for chunk in chunks]

    vector_store = QdrantVectorStore(client=client, collection_name=name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(docs, storage_context=storage_context)


def load_index(project_name, client=None):
    """Load an ALREADY-BUILT index straight from Qdrant - no README
    fetch, no embedding calls, just wrapping the existing collection in
    a VectorStoreIndex object. This is what mcp_server.py and web_app.py
    actually call at startup now, not build_index()."""
    configure_llama_index()
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    client = client or get_qdrant_client()
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name(project_name))
    return VectorStoreIndex.from_vector_store(vector_store)


def build_all_indexes(force=False):
    """Provisions every project's index - the real, expensive work. This
    is what 01_indexing.ipynb calls, not the live services. force=False
    (the default) skips any project whose collection already exists, so
    re-running the notebook after adding one new project only embeds
    that one project, not all 11 again. force=True rebuilds everything
    (e.g. after editing a README and wanting the index to reflect the
    change)."""
    client = get_qdrant_client()
    indexes = {}
    for name in config.PROJECTS:
        if not force and index_exists(name, client):
            print(f"{name}: already indexed, skipping (pass force=True to rebuild)")
            indexes[name] = load_index(name, client)
            continue
        print(f"Indexing {name}...")
        indexes[name] = build_index(name, client)
    return indexes


def load_all_indexes():
    """What mcp_server.py and web_app.py actually call at startup - fast,
    free, no embedding calls. Raises a clear, actionable error if the
    persistent index hasn't been provisioned yet, instead of silently
    falling back to an expensive rebuild."""
    client = get_qdrant_client()
    missing = [name for name in config.PROJECTS if not index_exists(name, client)]
    if missing:
        raise RuntimeError(
            f"No persistent index found for: {', '.join(missing)}. "
            "Run notebooks/01_indexing.ipynb first (with QDRANT_URL set to a real "
            "Qdrant Cloud instance) to provision it - see the README's "
            "'Provisioning the index' section."
        )
    return {name: load_index(name, client) for name in config.PROJECTS}


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class ProjectRouter:
    """Picks which project(s) a question is about, by comparing the
    question's embedding to each project's description embedding.
    No LLM call involved -- just embedding similarity."""

    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.names = list(config.PROJECT_DESCRIPTIONS.keys())
        self.description_embeddings = [
            embed_model.get_text_embedding(config.PROJECT_DESCRIPTIONS[name])
            for name in self.names
        ]

    def scores(self, question):
        """(project_name, similarity_score) for every project, best first."""
        q_embedding = self.embed_model.get_text_embedding(question)
        scored = [
            (name, cosine_similarity(q_embedding, emb))
            for name, emb in zip(self.names, self.description_embeddings)
        ]
        scored.sort(key=lambda pair: -pair[1])
        return scored

    def select(self, question):
        """The project name(s) this question should be answered from."""
        scored = self.scores(question)
        picked = [name for name, score in scored if score >= config.SIMILARITY_THRESHOLD]
        picked = picked[:config.MAX_PROJECTS_PER_QUERY]
        return picked if picked else [scored[0][0]]  # always return at least one guess


def build_router():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
    return ProjectRouter(embed_model)


class AgentState(TypedDict, total=False):
    question: str
    target_projects: list
    context: str
    draft: str
    passed: bool
    feedback: str
    retries: int
    final_answer: str


def build_agent(indexes, router):
    """The agent: plan which project(s) to use, retrieve, check the
    draft answer is actually grounded in what was retrieved, and retry
    (with the critique's feedback folded in) if it isn't."""
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0.2)

    def plan(state):
        projects = router.select(state["question"])
        projects = [p for p in projects if p in indexes]
        if not projects:
            projects = list(indexes.keys())[:1]
        return {"target_projects": projects}

    def retrieve(state):
        query = state["question"]
        if state.get("feedback"):
            query += f" (also cover: {state['feedback']})"

        parts = []
        for name in state["target_projects"]:
            engine = indexes[name].as_query_engine(similarity_top_k=5)
            response = engine.query(query)
            parts.append(f"From the {name} project:\n{response}")
        context = "\n\n".join(parts)

        prompt = (
            "Answer the question using only the context below. Write in "
            "plain prose - don't cite sources with bracket notation like "
            "[project_name], and don't repeat project names as labels; "
            "the interface already shows which project(s) this answer "
            "came from separately, so just answer naturally, the way "
            "you'd explain it to someone who already knows what they "
            "asked about. If the context isn't enough, say so instead "
            "of guessing.\n\n"
            f"Context:\n{context}\n\nQuestion: {state['question']}\n\nAnswer:"
        )
        draft = _extract_text(llm.invoke(prompt))
        return {"context": context, "draft": draft}

    def critique(state):
        prompt = (
            "Is this answer actually supported by the context, with nothing "
            "made up?\n\n"
            f"Context:\n{state['context']}\n\nAnswer:\n{state['draft']}\n\n"
            "Reply with exactly one line: PASS or FAIL: <reason>"
        )
        result = _extract_text(llm.invoke(prompt))
        passed = result.strip().upper().startswith("PASS")
        feedback = None if passed else (result.split(":", 1)[1].strip() if ":" in result else result)
        return {
            "passed": passed,
            "feedback": feedback,
            "retries": state.get("retries", 0) + (0 if passed else 1),
        }

    def after_critique(state):
        if state["passed"] or state["retries"] >= config.MAX_CRITIQUE_RETRIES:
            return "answer"
        return "retrieve"

    def answer(state):
        return {"final_answer": state["draft"]}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("critique", critique)
    graph.add_node("answer", answer)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "critique")
    graph.add_conditional_edges("critique", after_critique, {"retrieve": "retrieve", "answer": "answer"})
    graph.add_edge("answer", END)

    return graph.compile(checkpointer=MemorySaver())


def ask(agent, question, thread_id="default"):
    """Run the agent on one question, return the answer plus how it got there."""
    run_config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"question": question, "retries": 0}, run_config)
    return {
        "answer": result["final_answer"],
        "projects": result["target_projects"],
        "retries": result["retries"],
        "llm_calls": (result["retries"] + 1) * 2,  # one draft + one critique per attempt
    }


def naive_ask(index, question):
    """The baseline: one retrieval, one answer, no checking at all."""
    engine = index.as_query_engine(similarity_top_k=5)
    response = engine.query(question)
    return {"answer": str(response), "llm_calls": 1}
