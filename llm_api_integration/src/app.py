"""FastAPI app — routes only, no business logic.

Everything here is a thin call into client.py / tools.py / schemas.py,
each of which works and is testable as plain Python on its own. FastAPI
is just an HTTP shell around logic that already works; building the API
layer first would mean debugging the logic and the web framework at the
same time, for no reason.
"""

import json
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Must run before build_model() reads GEMINI_API_KEY from os.environ —
# otherwise .env sits on disk unused and the app fails with a confusing
# "key not set" error despite the file existing right next to it.
load_dotenv()

from src.client import build_model, call_gemini, stream_gemini, strip_markdown_fences
from src.config import load_config
from src.schemas import SentimentResult, ToolCallRequest
from src.tools import run_tool
from src.tracking import extract_token_counts, init_tracking, log_usage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()

# Applies mlflow.tracking_uri / experiment_name from config.yaml — without
# this call, MLflow silently falls back to its default ./mlruns folder and
# "Default" experiment, ignoring the config entirely.
init_tracking(config["mlflow"]["tracking_uri"], config["mlflow"]["experiment_name"])

# Two model instances: one free-text (streaming, natural prose), one forced
# into JSON mode (structured output, tool routing). Mixing json_mode into a
# single instance would make every prose response come back as JSON too.
model = build_model(
    model_name=config["model"]["name"],
    temperature=config["model"]["temperature"],
    max_output_tokens=config["model"]["max_output_tokens"],
)
json_model = build_model(
    model_name=config["model"]["name"],
    temperature=config["model"]["temperature"],
    max_output_tokens=config["model"]["max_output_tokens"],
    json_mode=True,
)

app = FastAPI(title="LLM API Integration")

@app.get("/health")
def health():
    return {"status": "ok", "model": config["model"]["name"]}

class AnalyzeRequest(BaseModel):
    text: str


class ChatRequest(BaseModel):
    prompt: str


class ToolChatRequest(BaseModel):
    prompt: str


@app.post("/analyze", response_model=SentimentResult)
def analyze(request: AnalyzeRequest) -> SentimentResult:
    """Return sentiment as validated structured JSON."""
    instruction = (
        "Analyze the sentiment of the following text. "
        "Respond ONLY with JSON matching this shape: "
        '{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0, "reasoning": "..."}. '
        f"Text: {request.text}"
    )
    response = call_gemini(
        json_model,
        instruction,
        max_attempts=config["retry"]["max_attempts"],
        backoff_seconds=config["retry"]["backoff_seconds"],
    )

    prompt_tokens, response_tokens = extract_token_counts(response)
    log_usage(
        prompt_tokens,
        response_tokens,
        config["model"]["name"],
        input_cost_per_million=config["pricing"]["input_cost_per_million"],
        output_cost_per_million=config["pricing"]["output_cost_per_million"],
    )

    try:
        cleaned = strip_markdown_fences(response.text)
        return SentimentResult.model_validate_json(cleaned)
    except Exception as e:
        # Surfacing this as a 502 (bad upstream response), not a 500 —
        # it's Gemini's output that failed validation, not our own bug.
        raise HTTPException(status_code=502, detail=f"Model returned invalid JSON: {e}")


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Stream the response back token-by-token as it's generated.

    No retry here, unlike call_gemini — if a chunk has already been sent to
    the client, retrying from scratch would duplicate output they've
    already seen. Instead we fail the stream cleanly: log what happened and
    stop, rather than let an unhandled exception break the connection mid-
    response (which is what happens by default if a chunk is safety-
    filtered — accessing chunk.text on a chunk with no candidates raises).
    """
    def token_generator():
        last_chunk = None
        try:
            for chunk in stream_gemini(model, request.prompt):
                last_chunk = chunk
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.warning(f"Stream interrupted: {e}")
            yield "\n[response interrupted]"
            return

        try:
            if last_chunk is not None:
                prompt_tokens, response_tokens = extract_token_counts(last_chunk)
                log_usage(
                    prompt_tokens,
                    response_tokens,
                    config["model"]["name"],
                    input_cost_per_million=config["pricing"]["input_cost_per_million"],
                    output_cost_per_million=config["pricing"]["output_cost_per_million"],
                )
        except Exception as e:
            logger.warning(f"Usage logging failed for streamed response: {e}")

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.post("/chat/tools")
def chat_with_tools(request: ToolChatRequest):
    """Let Gemini decide whether to call a tool, run it, then return the final answer.

    Simplified for clarity: real Gemini function-calling wires tool
    schemas into the model call itself. Here we ask the model to emit a
    JSON tool-call decision, execute it ourselves, then ask it to answer
    using the tool's result — same control flow, fewer moving parts to
    explain.
    """
    routing_prompt = (
        "Decide if you need a tool to answer this. Available tools: "
        "get_current_weather(city), search_documents(query). "
        'If yes, respond ONLY with JSON: {"tool_name": "...", "arguments": {...}}. '
        'If no tool is needed, respond ONLY with JSON: {"tool_name": null, "arguments": {}}. '
        f"User request: {request.prompt}"
    )
    routing_response = call_gemini(
        json_model,
        routing_prompt,
        max_attempts=config["retry"]["max_attempts"],
        backoff_seconds=config["retry"]["backoff_seconds"],
    )

    # Logged immediately, not batched at the end — this call already
    # happened and cost real tokens regardless of what we do with the
    # result next. If we waited until after decision-parsing/tool-execution
    # succeeded, a failure in either would silently drop this usage record.
    prompt_tokens, response_tokens = extract_token_counts(routing_response)
    log_usage(
        prompt_tokens,
        response_tokens,
        config["model"]["name"],
        input_cost_per_million=config["pricing"]["input_cost_per_million"],
        output_cost_per_million=config["pricing"]["output_cost_per_million"],
    )

    try:
        decision = ToolCallRequest.model_validate_json(strip_markdown_fences(routing_response.text))
    except (json.JSONDecodeError, ValueError) as e:
        # ValueError covers pydantic's ValidationError (a subclass of it),
        # so one except clause handles "not JSON at all" and "JSON but
        # wrong shape" the same way — both are the model's fault.
        raise HTTPException(status_code=502, detail=f"Model returned invalid routing decision: {e}")

    if not decision.tool_name:
        return {"answer": routing_response.text, "tool_used": None, "tool_result": None}

    try:
        tool_result = run_tool(decision.tool_name, decision.arguments)
    except (ValueError, TypeError) as e:
        # ValueError: unknown tool name. TypeError: model passed the wrong
        # argument names/shape for the tool. Both are the model's fault,
        # not ours — 502, same reasoning as the JSON validation error above.
        raise HTTPException(status_code=502, detail=f"Tool call failed: {e}")

    final_prompt = (
        f"User asked: {request.prompt}\n"
        f"Tool '{decision.tool_name}' returned: {tool_result}\n"
        "Answer the user's request using this information."
    )
    final_response = call_gemini(
        model,
        final_prompt,
        max_attempts=config["retry"]["max_attempts"],
        backoff_seconds=config["retry"]["backoff_seconds"],
    )

    prompt_tokens, response_tokens = extract_token_counts(final_response)
    log_usage(
        prompt_tokens,
        response_tokens,
        config["model"]["name"],
        input_cost_per_million=config["pricing"]["input_cost_per_million"],
        output_cost_per_million=config["pricing"]["output_cost_per_million"],
    )

    return {"answer": final_response.text, "tool_used": decision.tool_name, "tool_result": tool_result}


if __name__ == "__main__":
    # Lets `python -m src.app` honor config.yaml's serving.host/port.
    # `uvicorn src.app:app --reload --port 8000` (the dev workflow in the
    # README) bypasses this entirely and takes the port from the CLI flag
    # instead — that's fine for local dev with --reload, but this entrypoint
    # is what actually makes the "serving" config section in config.yaml
    # mean something, rather than sitting there unread.
    import uvicorn

    uvicorn.run(app, host=config["serving"]["host"], port=config["serving"]["port"])
