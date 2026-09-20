"""Run once at Docker build time. Downloads the GGUF model and the embedding
model into the image, so a container restart doesn't re-download ~GBs from
Hugging Face before it can answer anything.

Reads the same repo/filename as src/config.py (env vars override the defaults).
"""
import os

repo = os.environ.get("GGUF_REPO", "hossam3759180/support-copilot-gguf")
filename = os.environ.get("GGUF_FILENAME", "support-copilot-q4.gguf")

from huggingface_hub import hf_hub_download

print(f"Downloading {repo}/{filename} ...")
print("GGUF cached at:", hf_hub_download(repo_id=repo, filename=filename))

from sentence_transformers import SentenceTransformer

print("Downloading embedding model all-MiniLM-L6-v2 ...")
SentenceTransformer("all-MiniLM-L6-v2")
print("Models baked into the image.")
