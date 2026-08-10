# Deploys impl_vanilla/src/api.py as a Hugging Face Space (Docker SDK) --
# a real, persistent URL, unlike running `uvicorn` inside a Colab cell
# (which dies the moment that session ends and was never reachable from
# outside Colab in the first place).
#
# Setup:
#   1. huggingface.co -> New Space -> Docker SDK, point it at this repo
#      (or push this rag_chatbot/ folder to the Space's own git remote).
#   2. Space -> Settings -> Repository secrets, set:
#        GEMINI_API_KEY, QDRANT_API_KEY
#      (never bake these into the image -- Spaces secrets are injected as
#      env vars at runtime, same as everywhere else in this repo.)
#   3. configs/config.yaml's qdrant.url must point at a cluster that
#      already has data -- run impl_vanilla/notebooks/04_evaluation.ipynb
#      at least once first (see that notebook's Qdrant Cloud setup cell).
#
# Spaces expects the container to listen on port 7860 (configs/config.yaml's
# serving.port) -- that's the one Spaces-specific constraint here, nothing
# else in api.py is Spaces-specific.

FROM python:3.11-slim

WORKDIR /app

# configs/ and shared/ are one level up from impl_vanilla/ in the repo --
# copy the whole rag_chatbot/ folder in, not just impl_vanilla/, so those
# relative imports (api.py's _CONFIG_PATH, shared.llm_client, etc.) resolve
# the same way they do everywhere else in this repo.
COPY . /app

RUN pip install --no-cache-dir -r impl_vanilla/requirements.txt

EXPOSE 7860

# api.py's own imports are `from src.generation import ...` (needs
# impl_vanilla/ on the path) and generation.py's are `from shared.llm_client
# import ...` (needs the repo root on the path) -- same two-path pattern
# every notebook sets up with sys.path.append(base); sys.path.append(repo_root).
# `--app-dir impl_vanilla` alone only covers the first one and left `shared`
# unresolvable; caught by actually simulating uvicorn's import resolution
# with only impl_vanilla on the path before shipping this.
ENV PYTHONPATH=/app/impl_vanilla:/app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
