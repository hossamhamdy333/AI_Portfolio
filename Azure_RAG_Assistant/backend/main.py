import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from azure_storage import upload_to_blob_storage
from text_processing import process_and_upsert
from agent import run_agent
from config import settings, logger

app = FastAPI(title="Azure RAG Assistant API")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "static"))

# CORS: credentials + wildcard origins is invalid/insecure, so we don't allow
# credentials here. Tighten allow_origins to your real frontend URL(s) in
# production instead of "*". Since the UI is now served from this same
# backend (see "/" below), CORS mainly matters if you call the API from a
# separate origin (e.g. a custom frontend you build later).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    """
    Simple shared-secret check. If APP_API_KEY is unset, auth is disabled
    (convenient for local-only testing, NOT recommended for a public URL).
    """
    if settings.APP_API_KEY and x_api_key != settings.APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
    return True


class ChatRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the built-in single-page chat UI. No separate frontend
    deployment needed - visiting the backend's own URL is the app."""
    return templates.TemplateResponse(
        request, "index.html", {"app_api_key": settings.APP_API_KEY}
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        answer = run_agent(request.query)
        return {"answer": answer}
    except Exception as e:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        chunks = process_and_upsert(file_bytes, file.filename)
        blob_url = upload_to_blob_storage(file_bytes, file.filename)  # best-effort, may be None

        return {"status": "success", "blob_url": blob_url, "chunks": chunks}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))
