import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chroma_db_dir: str = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "documind")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 800))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 120))
    top_k: int = int(os.getenv("TOP_K", 4))


settings = Settings()
