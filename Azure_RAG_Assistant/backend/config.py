import logging
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-3.1-flash-lite"

    HF_TOKEN: str
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION_NAME: str = "azure-rag-assistant"

    AZURE_STORAGE_CONNECTION_STRING: str = ""
    AZURE_STORAGE_CONTAINER_NAME: str = "omnirag-documents"

    # Observability (Arize Phoenix). Off by default so tests and local
    # runs never require a running Phoenix collector.
    ENABLE_OBSERVABILITY: bool = False
    PHOENIX_COLLECTOR_ENDPOINT: str = ""

    # Inference backend: "gemini" (default, API) or "local" (self-hosted,
    # via Ollama). Point LOCAL_LLM_BASE_URL at Ollama's own OpenAI-compatible
    # endpoint (http://localhost:11434/v1 once `ollama serve` is running and
    # you've pulled a model with `ollama pull llama3.1`). See agent.py's
    # get_llm() and scripts/load_test.py for the comparison.
    LLM_BACKEND: str = "gemini"
    LOCAL_LLM_BASE_URL: str = "http://localhost:11434/v1"
    LOCAL_LLM_MODEL: str = "llama3.1"

    # Approximate cost, USD per million tokens - used only to print a
    # rough cost-per-1000-queries estimate in the load test's output.
    GEMINI_INPUT_COST_PER_MILLION: float = 0.075
    GEMINI_OUTPUT_COST_PER_MILLION: float = 0.30

    # Database. sqlite:///./dev.db for local dev/testing (zero setup);
    # mssql+pymssql://... for Azure SQL in production - see database.py
    # and README's "Setting up Azure SQL" section.
    DATABASE_URL: str = "sqlite:///./dev.db"

    # Rate limits, enforced per-user (chat/upload) or per-IP (auth, since
    # there's no logged-in user yet at that point) - see rate_limit.py.
    # Defaults are deliberately conservative since every user shares one
    # Gemini API key/quota; there is no per-user billing to fall back on.
    # Off by default in tests (set RATE_LIMIT_ENABLED=false) since a test
    # suite that legitimately calls /auth/register a dozen times in a row
    # would otherwise trip the same limit meant to catch real abuse - and
    # for the same reason, many real users sharing one IP (behind a
    # corporate NAT/VPN) should not be penalized as if they were one
    # attacker; per-account limits (chat/upload) don't have this problem.
    RATE_LIMIT_ENABLED: bool = True
    CHAT_RATE_LIMIT_PER_HOUR: int = 30
    UPLOAD_RATE_LIMIT_PER_HOUR: int = 10
    LOGIN_RATE_LIMIT_PER_15MIN: int = 10
    REGISTER_RATE_LIMIT_PER_HOUR: int = 5

    # Auth. JWT_SECRET_KEY MUST be a real random secret in production -
    # generate one with `python -c "import secrets; print(secrets.token_hex(32))"`
    # and never commit it. Anyone who has this value can forge a valid
    # access token for any user, including an admin.
    JWT_SECRET_KEY: str = "dev-only-secret-change-this-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 20
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # Google OAuth - leave blank to disable the "Continue with Google"
    # button; email+password login still works either way.
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/auth/google/callback"
    FRONTEND_URL: str = "http://localhost:8000"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AzureRAGAssistant-Backend")

settings = Settings()

class ResilientEmbeddings(Embeddings):
    """Wraps the Hugging Face embedding endpoint with three protections.

    The hosted endpoint is a free, shared service: it answers 429 (rate
    limited) or 503 (model still loading) from time to time, and it can
    reject a single huge request. Without this, one such answer turned into
    a 500 on chat, or a failed upload of a large document.

      - retries with growing pauses (1s, 2s, 4s) before giving up,
      - documents are embedded in batches instead of one giant request,
      - several batches run at once, which makes indexing a large file
        noticeably faster (it is network-bound, not CPU-bound).

    Results always come back in the original order.
    """

    def __init__(self, inner, batch_size: int = 32, max_workers: int = 4, attempts: int = 4, base_delay: float = 1.0):
        self._inner = inner
        self._batch_size = batch_size
        self._max_workers = max_workers
        self._attempts = attempts
        self._base_delay = base_delay

    def _with_retry(self, fn, *args):
        for attempt in range(1, self._attempts + 1):
            try:
                return fn(*args)
            except Exception as exc:
                if attempt == self._attempts:
                    raise
                pause = self._base_delay * (2 ** (attempt - 1))
                logger.warning("Embedding request failed (attempt %d/%d), retrying in %.0fs: %s",
                               attempt, self._attempts, pause, str(exc)[:160])
                time.sleep(pause)

    def embed_query(self, text: str) -> list[float]:
        return self._with_retry(self._inner.embed_query, text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        batches = [texts[i:i + self._batch_size] for i in range(0, len(texts), self._batch_size)]
        if len(batches) == 1:
            return self._with_retry(self._inner.embed_documents, batches[0])
        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(batches))) as pool:
            results = list(pool.map(lambda batch: self._with_retry(self._inner.embed_documents, batch), batches))
        return [vector for batch_result in results for vector in batch_result]


embeddings = ResilientEmbeddings(
    HuggingFaceEndpointEmbeddings(
        model=settings.HF_EMBEDDING_MODEL,
        huggingfacehub_api_token=settings.HF_TOKEN,
    )
)
