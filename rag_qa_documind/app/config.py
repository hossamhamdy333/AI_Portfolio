import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
<<<<<<< HEAD
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
=======
    """
    gemini_api_key and gemini_model are read dynamically via properties
    (not fixed at class-definition time) so that if something sets
    os.environ AFTER this module is first imported -- e.g. Streamlit Cloud
    injecting secrets into os.environ slightly after app startup -- the
    correct value is still picked up on the next call, without needing a
    full process restart.
    """

    @property
    def gemini_api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", "")

    @property
    def gemini_model(self) -> str:
        return os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

>>>>>>> 1daa2d74e09f7db542620d4ab4861f9cf5e0dc25
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chroma_db_dir: str = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "documind")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 800))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 120))
    top_k: int = int(os.getenv("TOP_K", 4))


<<<<<<< HEAD
settings = Settings()
=======
settings = Settings()
>>>>>>> 1daa2d74e09f7db542620d4ab4861f9cf5e0dc25
