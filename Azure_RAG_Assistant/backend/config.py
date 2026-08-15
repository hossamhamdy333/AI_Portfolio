import logging
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

    # Shared-secret header check between frontend and backend.
    # Leave blank to disable (fine for local-only testing, not for a public URL).
    APP_API_KEY: str = ""


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AzureRAGAssistant-Backend")

settings = Settings()

embeddings = HuggingFaceEndpointEmbeddings(
    model=settings.HF_EMBEDDING_MODEL,
    huggingfacehub_api_token=settings.HF_TOKEN,
)
