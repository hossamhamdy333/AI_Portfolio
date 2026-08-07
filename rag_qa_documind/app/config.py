import os
from pathlib import Path
from dotenv import load_dotenv
# load_dotenv() isn't guaranteed to search upward and find .env in that case.
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

class Settings:

    @property
    def gemini_api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", "")

    @property
    def gemini_model(self) -> str:
        return os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chroma_db_dir: str = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "documind")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 800))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 120))
    top_k: int = int(os.getenv("TOP_K", 4))


settings = Settings()
