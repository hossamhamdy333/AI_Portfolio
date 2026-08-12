---
title: AI Support Copilot
emoji: 🎧
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.6.0
app_file: gradio_app.py
pinned: false
---

# AI Support Copilot

A RAG-grounded customer support chatbot: a QLoRA-fine-tuned Llama-3 (8B)
model, retrieval over a knowledge base built from the Bitext customer
support dataset, with an optional Gemini-based faithfulness evaluator.

> The YAML block above is required by Hugging Face Spaces (it's how HF
> knows this is a Gradio app and which file to run). It's harmless on
> GitHub -- it just renders as a small metadata block at the top of the
> page.

## Two deployment targets, on purpose

This repo supports two different ways to run the model, because they solve
different problems:

| | `gradio_app.py` | `src/app.py` |
|---|---|---|
| SDK / framework | Gradio | FastAPI |
| Where it runs | **Hugging Face Spaces (free tier)** | Any host with a real, persistent GPU (e.g. an Azure VM/Container App) |
| GPU | HF **ZeroGPU** -- shared, free, allocated per-request via `@spaces.GPU` | Whatever GPU the host gives you |
| Frontend | Gradio's built-in chat UI | `frontend/index.html` (plain HTML/JS) |
| Cost | Free, no card | Depends on the host |

**Important:** ZeroGPU only schedules onto **Gradio SDK** Spaces -- Docker
and Static Spaces cannot use it. That's why `gradio_app.py`, not
`src/app.py`, is what you deploy to Hugging Face Spaces. `src/app.py` and
the `Dockerfile` are there for later, once you have a platform with an
always-on GPU (e.g. your Azure for Students credits).

## Architecture

```
User -> Gradio UI (gradio_app.py) -> KBRetriever (src/retriever.py) -> vector search
                    |
                    v
      Fine-tuned Llama-3 LoRA adapter (from HF Hub, loaded once at startup)
                    |
                    v
      generate() wrapped in @spaces.GPU -- claims a shared GPU slot only
      for the few seconds each request needs it
```

- **`src/prepare_data.py`** — downloads the Bitext dataset, formats it into
  `<|system|>/<|user|>/<|assistant|>` training examples, and builds
  `data/kb_articles.jsonl` (the retrieval knowledge base).
- **`notebooks/training.ipynb`** — Colab notebook that fine-tunes
  `unsloth/llama-3-8b-bnb-4bit` with QLoRA, then pushes the adapter to HF Hub.
- **`src/retriever.py`** — embeds KB articles with `sentence-transformers`
  and indexes them in an in-memory Chroma collection for similarity search.
  Runs on CPU either way, so it doesn't touch the GPU budget.
- **`gradio_app.py`** — the HF Spaces entry point. Loads the model at
  module scope (the documented ZeroGPU pattern), decorates the generation
  call with `@spaces.GPU`.
- **`src/app.py`** + **`Dockerfile`** + **`frontend/index.html`** — a
  self-contained FastAPI + static-HTML stack for deploying somewhere with
  a real GPU later.
- **`src/evaluate.py`** — Gemini-based faithfulness check (is the response
  actually supported by the retrieved context, or did it hallucinate).

## Local setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Build training data + KB
python src/prepare_data.py

# 2. Fine-tune on Colab (GPU) -- run notebooks/training.ipynb top to bottom.
#    The last cell pushes the adapter to Hugging Face Hub.

# 3a. Run the Gradio app locally (same thing that runs on HF Spaces,
#     minus real ZeroGPU -- needs a local GPU or will error on .to("cuda"))
cp .env.example .env
python gradio_app.py

# 3b. OR run the FastAPI + HTML version locally (works on CPU too, just slow)
uvicorn src.app:app --reload --port 8000
# open frontend/index.html directly, or serve it separately
```

## Deploying to Hugging Face Spaces (free, no card required)

1. Create a free account at huggingface.co if you don't have one.
2. Create a new Space: profile icon (top right) → **New Space**.
   - **SDK: Gradio** (this is the important part — not Docker)
   - Hardware: default (free CPU basic) is fine to select; ZeroGPU access
     comes from the `@spaces.GPU` decorator in code, not a hardware tier
     you pick manually. If your account has ZeroGPU enabled, the Space
     will request it automatically at call time.
3. Push this repo's contents to the Space's git remote (Spaces are git
   repos):
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
4. In the Space, go to **Settings → Variables and secrets** and add:
   - `MODEL_REPO` = `hossamhamdy333/support-copilot-llama3-lora`
   - `GEMINI_API_KEY` and `ENABLE_EVAL=1` (only if you want the inline faithfulness check active — it costs one extra Gemini call per chat message)
5. Watch the build logs. First boot is slow (2–5 min, downloading the base
   model + adapter). Once you see `Model loaded.` and `Retriever ready.`,
   the Space's public URL is your live demo link for the CV.

### Notes / constraints
- `data/kb_articles.jsonl` must be committed to the repo (run
  `prepare_data.py` locally first) — the container doesn't regenerate it.
- Free-tier Spaces have no uptime guarantee and can sleep after inactivity
  — fine for a portfolio demo, not for 24/7 production.
- ZeroGPU has a daily quota per account; heavy testing can exhaust it
  temporarily (requests will queue or fail until it resets).

## Environment variables

See `.env.example`.
