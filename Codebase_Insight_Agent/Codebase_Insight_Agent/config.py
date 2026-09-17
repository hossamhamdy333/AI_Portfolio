# Settings shared by every notebook and by mcp_server.py.
# Plain constants on purpose -- no config file format, nothing to parse.

GITHUB_ORG = "hossamhamdy333"
GITHUB_REPO = "AI_Portfolio"
GITHUB_BRANCH = "main"

LLM_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Persistent vector storage (Qdrant Cloud - free tier is enough for this).
# Leave blank to fall back to a local in-memory Qdrant instead, but that
# mode has no persistence at all - every process restart rebuilds every
# index from scratch. See portfolio.py's get_qdrant_client() and the
# README's "Provisioning the index" section.
import os as _os
QDRANT_URL = _os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = _os.environ.get("QDRANT_API_KEY", "")

SIMILARITY_THRESHOLD = 0.3   # a project below this score is not considered relevant
MAX_PROJECTS_PER_QUERY = 3   # how many project indexes one question can be routed to
MAX_CRITIQUE_RETRIES = 2     # how many times the agent can retry after a failed critique

PROJECTS = [
    "customer_support_copilot",
    "Azure_RAG_Assistant",
    "Credit_Fraud_Detection",
    "House_Price_Prediction",
    "allam_finetune",
    "customer_churn_prediction",
    "fact_check_crew",
    "rag_router",
    "rag-vanilla-vs-langchain",
    "nl2sql_finetune",
    "ecommerce-demand-forecasting",
    "employee-attrition",
    "llm_api_integration",
    "marketing-ab-testing",
    "rag_qa_documind",
    "semantic-search-arxiv-papers",
    "sentiment_forge",
]

# One short description per project, used by the router to match a
# question to the right project(s). Keep these specific.
PROJECT_DESCRIPTIONS = {
    "customer_support_copilot": "RAG customer support chatbot with a fine-tuned Llama-3-8B model (GGUF), deployed on Azure Container Apps",
    "Azure_RAG_Assistant": "Document-upload RAG chat assistant deployed on Azure",
    "Credit_Fraud_Detection": "Credit card fraud detection model, imbalanced data, SHAP explainability",
    "House_Price_Prediction": "House price prediction regression model",
    "allam_finetune": "QLoRA fine-tuning of an Arabic legal LLM, with before/after evaluation",
    "customer_churn_prediction": "Customer churn prediction classification model",
    "fact_check_crew": "Multi-agent pipeline that checks claims against retrieved sources",
    "rag_router": "Multi-domain RAG that routes questions between topic indexes",
    "rag-vanilla-vs-langchain": "Comparison of a vanilla RAG pipeline vs a LangChain implementation",
    "nl2sql_finetune": "Fine-tuning a small model for natural-language-to-SQL",
    "ecommerce-demand-forecasting": "Demand forecasting on the Online Retail II e-commerce dataset",
    "employee-attrition": "Employee attrition prediction model with a Postgres SQL analysis layer",
    "llm_api_integration": "FastAPI wrapper around an LLM API with retries, structured output, tool calling, and MLflow cost tracking, switchable between Gemini, Ollama, and vLLM backends",
    "marketing-ab-testing": "Statistical analysis of a marketing A/B test",
    "rag_qa_documind": "Document-upload RAG Q&A app with per-user accounts, deployed on Streamlit Cloud",
    "semantic-search-arxiv-papers": "Staged semantic search over arXiv ML papers (BM25 to SBERT/FAISS to Qdrant to cross-encoder reranking)",
    "sentiment_forge": "Sentiment analysis comparing TF-IDF, BiLSTM, and fine-tuned BERT models",
}

# --- web_app.py (the public recruiter-facing website) -----------------
# Visitors ask questions with no login at all - that's the point, zero
# friction for someone clicking a link on a CV. What protects the app
# instead of a login wall: a per-IP rate limit and the same guardrails
# layer used in Azure_RAG_Assistant / customer_support_copilot. The one
# thing that DOES need a login is the admin dashboard - just for you, to
# see what's being asked and what got blocked.
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./dev.db")

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-only-secret-change-this-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 20
REFRESH_TOKEN_EXPIRE_DAYS = 30

RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", 10))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", 3600))  # 1 hour
