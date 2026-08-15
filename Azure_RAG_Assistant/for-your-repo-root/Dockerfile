# Root-level Dockerfile for the AI_Portfolio monorepo.
#
# Azure App Service's "GitHub Actions" deploy wizard defaults to building
# from a Dockerfile at the repo root with the repo root as build context -
# it doesn't reliably expose a subfolder path option. Rather than hand-edit
# the auto-generated GitHub Actions workflow YAML (fragile, breaks if
# Azure's template changes), this file just lives at the repo root and
# points into Azure_RAG_Assistant/backend/ for everything it needs. This way
# the default wizard behavior works with zero extra configuration.
#
# This file only matters for the Azure deploy - it's a thin proxy to the
# real Dockerfile at Azure_RAG_Assistant/backend/Dockerfile, which is still
# the one used for local docker-compose development.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Azure_RAG_Assistant/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Azure_RAG_Assistant/backend/ .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
