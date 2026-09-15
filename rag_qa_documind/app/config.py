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

    # Database: sqlite:///./dev.db locally (zero setup), or Azure SQL in
    # production - see README's "Accounts and a real database" section.
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./dev.db")

    # Auth - generate with: python -c "import secrets; print(secrets.token_hex(32))"
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "dev-only-secret-change-this-in-production")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 20))
    refresh_token_expire_days: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 30))

    # Google OAuth - only used by the FastAPI backend's redirect-based
    # flow (app/oauth.py), not by the standalone streamlit_app.py, which
    # has no separate server to redirect back to - see that file's
    # docstring for why it's email+password only.
    google_client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
    google_client_secret: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    google_redirect_uri: str = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:8000")


settings = Settings()
