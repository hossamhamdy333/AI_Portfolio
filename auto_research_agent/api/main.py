"""
FastAPI backend wiring everything together.

Endpoints:
  POST /upload_knowledge   -> ingest a file (pdf/docx/xlsx/audio/video/image) into Qdrant
  POST /execute_task       -> run the full agent graph, returns final report
                               (auto-resumes past the human-review pause if no conflict found)
  POST /resume/{thread_id} -> resume a paused run after a human decision
  WS   /ws/stream          -> streams node-by-node progress for a task

Run: uvicorn api.main:app --reload --port 8000
"""
import os
import shutil
import uuid
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from langchain_groq import ChatGroq
from ingestion.chunking import get_qdrant, ingest_file
from agents.state import get_checkpointer, AgentState
from agents.graph import build_graph

app = FastAPI(title="Auto Research Agent API")

QDRANT_PATH = os.getenv("QDRANT_PATH", "./data/qdrant_db")
SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/agent_memory.sqlite")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
qdrant_client = get_qdrant(QDRANT_PATH)
checkpointer = get_checkpointer(SQLITE_PATH)
graph_app = build_graph(llm, qdrant_client, checkpointer)


class TaskRequest(BaseModel):
    task: str


class ResumeRequest(BaseModel):
    approve: bool
    note: str = ""


@app.post("/upload_knowledge")
async def upload_knowledge(file: UploadFile = File(...)):
    tmp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        n_chunks = ingest_file(qdrant_client, tmp_path)
    finally:
        os.remove(tmp_path)
    return {"filename": file.filename, "chunks_indexed": n_chunks}


@app.post("/execute_task")
async def execute_task(req: TaskRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    init_state: AgentState = {
        "task": req.task, "plan": "", "research": "", "analysis": "",
        "report": "", "needs_human_review": False, "review_note": None,
    }
    result = graph_app.invoke(init_state, config)

    if result.get("needs_human_review"):
        # Graph is paused before "analyst". Caller must hit /resume/{thread_id}.
        return {"status": "awaiting_human_review", "thread_id": thread_id,
                "review_note": result.get("review_note"), "research_so_far": result.get("research")}

    # No conflict flagged -> auto-resume past the interrupt to finish the run.
    result = graph_app.invoke(None, config)
    return {"status": "done", "thread_id": thread_id, "report": result["report"]}


@app.post("/resume/{thread_id}")
async def resume(thread_id: str, req: ResumeRequest):
    config = {"configurable": {"thread_id": thread_id}}
    if not req.approve:
        return {"status": "cancelled", "note": req.note}
    result = graph_app.invoke(None, config)
    return {"status": "done", "thread_id": thread_id, "report": result["report"]}


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """Streams which node is currently running, for a live 'agent thoughts' UI."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        init_state: AgentState = {
            "task": data["task"], "plan": "", "research": "", "analysis": "",
            "report": "", "needs_human_review": False, "review_note": None,
        }
        async for event in graph_app.astream(init_state, config):
            for node_name, node_state in event.items():
                await websocket.send_json({"node": node_name, "state": node_state})
        await websocket.send_json({"node": "DONE", "thread_id": thread_id})
    except WebSocketDisconnect:
        pass


@app.get("/health")
async def health():
    return {"status": "ok"}
