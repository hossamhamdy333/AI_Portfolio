"""
Lightweight RAG retriever for the Support Copilot.

Uses sentence-transformers for embeddings and Chroma (in-memory, ephemeral
client) as the vector store. Designed to be cheap enough to run on CPU so it
works fine inside a Hugging Face Spaces container without needing a GPU for
retrieval itself -- only generation (in app.py) needs the GPU.
"""

import json
import logging
import os

import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "kb_articles.jsonl")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class KBRetriever:
    """Embeds and indexes KB articles once, then answers similarity queries."""

    def __init__(self, kb_path: str = DEFAULT_KB_PATH, collection_name: str = "support_kb"):
        self.kb_path = kb_path
        logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL_NAME)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # In-memory Chroma client -- fine for a single-container Space.
        # If you outgrow this, swap for chromadb.PersistentClient(path=...).
        client = chromadb.Client()
        self.collection = client.get_or_create_collection(collection_name)

        if self.collection.count() == 0:
            self._index()
        else:
            logger.info("Collection '%s' already indexed (%d docs).", collection_name, self.collection.count())

    def _load_rows(self):
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(
                f"KB file not found at {self.kb_path}. "
                "Run `python src/prepare_data.py` first to generate it."
            )
        with open(self.kb_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _index(self):
        rows = self._load_rows()
        logger.info("Indexing %d KB articles...", len(rows))

        contents = [r["content"] for r in rows]
        embeddings = self.model.encode(contents, show_progress_bar=False).tolist()

        self.collection.add(
            ids=[str(i) for i in range(len(rows))],
            embeddings=embeddings,
            documents=contents,
            metadatas=[
                {"intent": r.get("intent", ""), "category": r.get("category", "")}
                for r in rows
            ],
        )
        logger.info("Indexing complete.")

    def retrieve(self, query: str, k: int = 1) -> str:
        """Return the top-k most relevant KB snippet(s) for a query, joined by newlines."""
        q_embedding = self.model.encode([query], show_progress_bar=False).tolist()
        results = self.collection.query(query_embeddings=q_embedding, n_results=k)

        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No relevant context found in the knowledge base."
        return "\n".join(docs)


# Simple manual smoke test: `python src/retriever.py "where is my refund"`
if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "where is my refund"
    retriever = KBRetriever()
    print(f"Query: {query}")
    print(f"Retrieved context:\n{retriever.retrieve(query)}")
