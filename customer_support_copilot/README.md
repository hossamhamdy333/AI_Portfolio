<div align="center">

# AI Support Copilot

A customer support chatbot powered by a QLoRA-fine-tuned Llama-3 (8B) model, grounded with RAG retrieval over a knowledge base built from the Bitext customer support dataset. Deployed live on Azure Container Apps, running entirely on CPU.

**Live demo:** [support-copilot-app...azurecontainerapps.io](https://support-copilot-app.blackpebble-352cd42a.francecentral.azurecontainerapps.io)

`Python` `FastAPI` `Llama-3-8B (QLoRA)` `llama-cpp-python (GGUF)` `Sentence-Transformers` `ChromaDB` `Docker`

</div>

---

### Contents

- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Running it locally](#running-it-locally)
- [Deploying](#deploying)
- [Why CPU instead of GPU](#why-cpu-instead-of-gpu)

## How it works

```
User message
    │
    ▼
FastAPI backend (src/app.py)
    │
    ├──► KBRetriever (src/retriever.py)
    │       embeds the query, finds the closest matching
    │       support article from the knowledge base
    │
    └──► llama.cpp (quantized GGUF model)
            generates a response grounded in that context
```

The model started as a QLoRA fine-tune of Llama-3-8B, trained on Colab and pushed to Hugging Face Hub. To make it run fast enough on CPU-only cloud hardware (no GPU available on the free Azure for Students tier), it was converted to GGUF format and quantized — that's what takes response times from "times out after 4 minutes" down to about 15-20 seconds.

## Project layout

| File | What it does |
|---|---|
| `src/prepare_data.py` | Downloads the Bitext dataset, builds the training set and the knowledge base (`data/kb_articles.jsonl`) |
| `notebooks/training.ipynb` | Fine-tunes Llama-3-8B with QLoRA on Colab, pushes the adapter to HF Hub |
| `notebooks/gguf_conversion.ipynb` | Converts the fine-tuned adapter to a quantized GGUF file (the format that runs fast on CPU) |
| `notebooks/eda.ipynb` | Exploratory analysis of the training dataset |
| `src/retriever.py` | Embeds and searches the knowledge base with `sentence-transformers` + Chroma |
| `src/app.py` | The FastAPI backend — loads the GGUF model with `llama-cpp-python`, retrieves context, generates answers |
| `frontend/index.html` | The chat UI, served directly by the backend |
| `src/evaluate.py` | Optional Gemini-based check for whether a response is actually supported by its retrieved context |
| `Dockerfile` | Builds the deployed image |

## Running it locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Build the knowledge base (only needed once)
python src/prepare_data.py

# Optional: copy .env.example to .env if you want to override the default
# model repo or enable the Gemini faithfulness check (ENABLE_EVAL=1)
cp .env.example .env

# Run the API + chat UI
uvicorn src.app:app --reload --port 8000
# open http://localhost:8000
```

## Deploying

This is deployed as a Docker container on **Azure Container Apps**, built and pushed via a GitHub Actions workflow (`.github/workflows/build-push-acr.yml`) rather than pushed from a local machine — this sidesteps large-image upload timeouts on a slow home connection.

Environment variables the container accepts (all optional, defaults shown work out of the box):

| Variable | Purpose |
|---|---|
| `GGUF_REPO` | Hugging Face repo holding the quantized model (default: `hossam3759180/support-copilot-gguf`) |
| `GGUF_FILENAME` | File name within that repo (default: `support-copilot-q4.gguf`) |
| `GEMINI_API_KEY` | Only needed if `ENABLE_EVAL=1` |
| `ENABLE_EVAL` | Set to `1` to run the faithfulness check on every response (costs one extra Gemini call per message) |

## Why CPU instead of GPU

Azure for Students doesn't include a GPU workload profile in this account's available regions. Rather than block the whole project on that, the model was converted from raw `transformers` inference (too slow on CPU — an 8B model in float32 would time out on every request) to a quantized GGUF file run through `llama.cpp`, which is specifically built for fast CPU inference.
