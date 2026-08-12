"""
FastAPI backend for the AI Support Copilot.

NOTE: this is NOT what runs on Hugging Face Spaces -- see gradio_app.py for
that (ZeroGPU only works with the Gradio SDK, not Docker/FastAPI). This app
is for deploying to a platform with a real, always-on GPU -- e.g. an Azure
VM or Container App once your student credits are active -- or for local
testing with your own GPU.

It loads the QLoRA-fine-tuned Llama-3 adapter from Hugging Face Hub,
retrieves relevant KB context for each query, and generates a grounded
response.
"""

import logging
import os

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from unsloth import FastLanguageModel

from src.retriever import KBRetriever
from src.evaluate import evaluate_faithfulness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_REPO = os.environ.get("MODEL_REPO", "hossamhamdy333/support-copilot-llama3-lora")
MAX_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are a senior customer support agent for a premium brand. "
    "Reply politely, professionally, and resolve the user's issue based "
    "ONLY on the provided context."
)

app = FastAPI(title="AI Support Copilot API")

# Populated at startup (see load_resources below).
_model = None
_tokenizer = None
_retriever = None


@app.on_event("startup")
def load_resources():
    """Load the model, tokenizer, and retriever once when the container boots."""
    global _model, _tokenizer, _retriever

    logger.info("Loading fine-tuned model from %s on device=%s ...", MODEL_REPO, DEVICE)
    if DEVICE != "cuda":
        logger.warning(
            "No CUDA device found -- running on CPU. Generation will be slow "
            "(minutes per response) for an 8B model. Fine for occasional testing, "
            "not recommended for a live demo."
        )
    _model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_REPO,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=(DEVICE == "cuda"),
    )
    FastLanguageModel.for_inference(_model)
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
    inputs = _tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=_tokenizer.eos_token_id,
        )
    decoded = _tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>")[-1]
    return decoded.strip()


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


# Serve the plain HTML/JS frontend (see frontend/index.html) at the root URL,
# so the whole app -- API + UI -- is one container, one Space.
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
