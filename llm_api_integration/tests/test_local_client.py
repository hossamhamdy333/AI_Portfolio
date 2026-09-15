"""
Tests local_client.py against a real, tiny HTTP server that speaks the
same OpenAI-compatible protocol Ollama/vLLM do - not just a mocked
requests.post, so the actual JSON parsing and SSE streaming logic is
genuinely exercised, not assumed to work.
"""

import json
import sys
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.local_client import build_model, call_local, stream_local


class _FakeOpenAICompatibleHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence the default per-request console logging

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for word in ["Hello", " there", "!"]:
                chunk = {"choices": [{"delta": {"content": word}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            usage_chunk = {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
            self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            response = {
                "choices": [{"message": {"content": "Hello, world!"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6},
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())


def _start_fake_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeOpenAICompatibleHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


def test_call_local_parses_a_real_response():
    server, port = _start_fake_server()
    try:
        model = build_model("test-model", temperature=0.7, max_output_tokens=100, base_url=f"http://127.0.0.1:{port}")
        response = call_local(model, "hi")
        assert response.text == "Hello, world!"
        assert response.usage_metadata.prompt_token_count == 4
        assert response.usage_metadata.candidates_token_count == 6
    finally:
        server.shutdown()


def test_stream_local_parses_real_sse_chunks():
    server, port = _start_fake_server()
    try:
        model = build_model("test-model", temperature=0.7, max_output_tokens=100, base_url=f"http://127.0.0.1:{port}")
        chunks = list(stream_local(model, "hi"))

        text_chunks = [c for c in chunks if c.text]
        full_text = "".join(c.text for c in text_chunks)
        assert full_text == "Hello there!"

        usage_chunks = [c for c in chunks if c.usage_metadata.prompt_token_count > 0]
        assert len(usage_chunks) == 1
        assert usage_chunks[0].usage_metadata.prompt_token_count == 5
        assert usage_chunks[0].usage_metadata.candidates_token_count == 3
    finally:
        server.shutdown()


def test_call_local_retries_on_failure_then_succeeds():
    """Same retry contract as call_gemini - point at a port nothing is
    listening on first, confirm it raises after exhausting retries."""
    model = build_model("test-model", temperature=0.7, max_output_tokens=100, base_url="http://127.0.0.1:1")
    import pytest
    with pytest.raises(RuntimeError):
        call_local(model, "hi", max_attempts=2, backoff_seconds=0)
