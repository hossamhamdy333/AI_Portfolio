FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-base.txt requirements-mcp.txt ./
RUN pip install --no-cache-dir -r requirements-mcp.txt

COPY config.py .
RUN python -c "import config; from sentence_transformers import SentenceTransformer; SentenceTransformer(config.EMBEDDING_MODEL)"

COPY data/ data/
COPY portfolio.py .
COPY mcp_server.py .

# GOOGLE_API_KEY, QDRANT_URL and QDRANT_API_KEY are set at deploy time.
# Without MCP_ACCESS_TOKEN the server has no auth, which is fine for local
# stdio use but not for a public URL (see README, "Remote deployment").
ENV PYTHONUNBUFFERED=1

# Only used when MCP_TRANSPORT is streamable-http or sse
EXPOSE 8000

CMD ["python", "mcp_server.py"]
