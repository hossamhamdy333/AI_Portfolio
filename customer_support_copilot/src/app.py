"""
FastAPI backend for the AI Support Copilot.

NOTE: this is NOT what runs on Hugging Face Spaces -- see gradio_app.py for
that. This app is for Azure Container Apps (or any CPU-only host).

Uses llama-cpp-python with a quantized GGUF model instead of raw
transformers/peft -- this is what actually makes CPU inference fast enough
to finish within Azure Container Apps' request timeout. The GGUF file was
produced by notebooks/gguf_conversion.ipynb (merges the LoRA adapter into
the base model at the GGUF level, then quantizes to Q4_K_M).

Loads the quantized GGUF model from Hugging Face Hub, retrieves relevant KB
context for each query, and generates a grounded response.
"""

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from src.retriever import KBRetriever
from src.evaluate import evaluate_faithfulness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GGUF_REPO = os.environ.get("GGUF_REPO", "hossam3759180/support-copilot-gguf")
GGUF_FILENAME = os.environ.get("GGUF_FILENAME", "support-copilot-q4.gguf")
MAX_NEW_TOKENS = 150
N_CTX = 1024

SYSTEM_PROMPT = (
    "You are a senior customer support agent for a premium brand. "
    "Reply politely, professionally, and resolve the user's issue based "
    "ONLY on the provided context."
)

app = FastAPI(title="AI Support Copilot API")

# Populated at startup (see load_resources below).
_model = None
_retriever = None


@app.on_event("startup")
def load_resources():
    """Download the GGUF model once and load it with llama-cpp-python."""
    global _model, _retriever

    logger.info("Downloading GGUF model from %s/%s ...", GGUF_REPO, GGUF_FILENAME)
    model_path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILENAME)

    logger.info("Loading model into llama.cpp ...")
    _model = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )
    logger.info("Model loaded.")

    logger.info("Building KB retriever...")
    _retriever = KBRetriever()
    logger.info("Retriever ready.")


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str
    context: str


def _generate(prompt: str) -> str:
    output = _model(
        prompt,
        max_tokens=MAX_NEW_TOKENS,
        stop=["<|user|>", "<|system|>"],
        temperature=0.0,
    )
    return output["choices"][0]["text"].strip()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if _model is None or _retriever is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")

    try:
        context = _retriever.retrieve(request.query)

        prompt = f"""<|system|>
{SYSTEM_PROMPT}
<|user|>
Context: {context}
Query: {request.query}
<|assistant|>
"""
        ai_text = _generate(prompt)

        if os.environ.get("ENABLE_EVAL") == "1":
            eval_result = evaluate_faithfulness(request.query, context, ai_text)
            if eval_result and not eval_result.get("is_faithful", True):
                logger.warning("Faithfulness check flagged this response: %s", eval_result.get("reason"))

        return ChatResponse(response=ai_text, context=context)

    except Exception as e:
        logger.exception("Chat generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
