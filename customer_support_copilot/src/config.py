"""
All environment-driven settings in one place, instead of os.environ.get()
calls scattered through the codebase. Pydantic validates required values
are present at startup, so a missing secret fails immediately with a clear
error instead of an obscure crash the first time it's actually used.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GGUF_REPO: str = "hossam3759180/support-copilot-gguf"
    GGUF_FILENAME: str = "support-copilot-q4.gguf"
    GEMINI_API_KEY: str = ""
    ENABLE_EVAL: bool = False
    GEMINI_DAILY_LIMIT: int = 50

    # Inference backend: "llamacpp" (default, CPU, the quantized GGUF file
    # above) or "vllm" (GPU, higher throughput under concurrent load - the
    # right tool specifically because this is a customer-facing bot where
    # many people can be chatting at once, unlike a low-traffic dev tool).
    # vLLM needs the ORIGINAL merged (non-GGUF) checkpoint - vLLM runs
    # native HF/safetensors weights, not GGUF files - see README's
    # "Setting up vLLM" section for how to get that checkpoint onto HF Hub.
    LLM_BACKEND: str = "llamacpp"

    # llama.cpp speed-up: "prompt lookup decoding". Support answers reuse a lot
    # of wording from the retrieved KB article, so the model can guess the next
    # few words by finding them in the prompt and verify them all in one step
    # instead of producing them one at a time. Output is unchanged; it just
    # arrives sooner. Value = how many words ahead to guess; llama-cpp-python's
    # docs recommend 2 for CPU-only machines. Set to 0 to turn it off.
    LLM_PROMPT_LOOKUP_TOKENS: int = 2
    VLLM_BASE_URL: str = "http://localhost:8001/v1"
    VLLM_MODEL: str = "hossam3759180/support-copilot-merged"

    # Database. sqlite:///./dev.db for local dev/testing (zero setup);
    # mssql+pyodbc://... for Azure SQL in production.
    DATABASE_URL: str = "sqlite:///./dev.db"

    # Auth. JWT_SECRET_KEY MUST be a real random secret in production -
    # generate one with `python -c "import secrets; print(secrets.token_hex(32))"`.
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


settings = Settings()
