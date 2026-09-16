FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY config.py .
COPY portfolio.py .
COPY mcp_server.py .

# GOOGLE_API_KEY must be provided at container run/deploy time -- never
# baked into the image. Same for MCP_ACCESS_TOKEN if MCP_TRANSPORT is set
# to streamable-http or sse (see README's "Remote deployment" section) --
# without MCP_ACCESS_TOKEN set, the server defaults to no auth at all,
# which is correct for local stdio use but not for a public URL.
ENV PYTHONUNBUFFERED=1

# Only used when MCP_TRANSPORT=streamable-http or sse - a plain local
# `python mcp_server.py` (stdio, the default) doesn't listen on a port at
# all, so this EXPOSE is a no-op for that case.
EXPOSE 8000

CMD ["python", "mcp_server.py"]
