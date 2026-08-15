import logging
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"

    HF_TOKEN: str
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "omnirag"

    AZURE_STORAGE_CONNECTION_STRING: str = ""
    AZURE_STORAGE_CONTAINER_NAME: str = "omnirag-documents"

    # Shared-secret header check between frontend and backend.
    # Leave blank to disable (fine for local-only testing, not for a public URL).
    APP_API_KEY: str = ""


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AzureRAGAssistant-Backend")

settings = Settings()

pc = Pinecone(api_key=settings.PINECONE_API_KEY)

embeddings = HuggingFaceEndpointEmbeddings(
    model=settings.HF_EMBEDDING_MODEL,
    huggingfacehub_api_token=settings.HF_TOKEN,
)
