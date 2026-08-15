# Azure RAG Assistant — Real Azure, Agentic RAG (Student Credits, No Card)

A single FastAPI service (API + a built-in web UI) that lets you upload
documents (PDF/image/txt), index them into a vector database, and chat with
an AI agent that can search those documents and do arithmetic.

**This version runs on real Microsoft Azure** — actual Azure Blob Storage
and an actual Azure hosting service — funded by your Azure for Students
credit ($100, no card required). This is genuine cloud infrastructure, not
a simulation, so it's fair to put "Azure" on your CV once it's deployed
this way.

**One deployment, one host.** The chat UI is a plain HTML/JS page served
directly by the backend at `/` — there's no separate frontend service to
deploy, no second account to manage.

## Stack

| Piece | Service | Why |
|---|---|---|
| LLM | Google Gemini (`gemini-2.5-flash`) | Free API, no card |
| Embeddings | Hugging Face Inference API | Free API, no card |
| Vector DB | Qdrant Cloud | Free forever tier, no card |
| Object storage | **Azure Blob Storage** (real) | Funded by your student credit |
| Hosting (API + UI, one service) | **Azure App Service** (Web App for Containers) | Funded by your student credit, matches the "Deploy a Docker container" / "Azure App Service" tiles on your portal |

Gemini, Hugging Face, and Qdrant stay as free third-party APIs regardless
of cloud provider — no reason to change those. The infrastructure pieces
(storage + hosting) are now real Azure instead of a free-tier workaround.

---

## 1. Get your free API keys (unrelated to Azure, still needed)

1. **Gemini**: [Google AI Studio](https://aistudio.google.com/apikey) → Create API key.
2. **Hugging Face**: [huggingface.co](https://huggingface.co) → Settings → Access Tokens → create a *Read* token.
3. **Qdrant**: [cloud.qdrant.io](https://cloud.qdrant.io) → sign up (no card) →
   **Create Cluster** → pick the **Free** tier → once it's up, copy the
   cluster URL and create/copy an API key from the **API Keys** tab.

## 2. Put the code on GitHub (browser only, no git needed)

1. Unzip `Azure_RAG_Assistant.zip`.
2. [github.com/new](https://github.com/new) → create a repo, e.g. `omnirag`.
3. On the repo page, click **"uploading an existing file"**, drag in
   everything from the unzipped folder, and commit.

## 3. Create a real Azure Storage Account (Blob Storage)

1. In the [Azure Portal](https://portal.azure.com), use the top search bar
   → search **"Storage accounts"** → click **+ Create**.
2. Fill in:
   - **Resource group**: click "Create new", name it `omnirag-rg`.
   - **Storage account name**: something globally unique, e.g. `omniragstorage123`.
   - **Region**: pick whatever's closest/default.
   - **Performance**: Standard. **Redundancy**: Locally-redundant storage (LRS) — cheapest, matches the free "5 GB LRS" tile on your portal.
3. Click **Review + create**, then **Create**. Wait ~1 minute for deployment.
4. Once it's ready, click **Go to resource**. In the left sidebar, under
   **Security + networking**, click **Access keys**. Click **Show** next to
   key1, and copy the **Connection string** value — this is your
   `AZURE_STORAGE_CONNECTION_STRING`.
5. Still in the storage account, left sidebar → **Data storage** →
   **Containers** → **+ Container**. Name it `omnirag-documents` (matches
   `AZURE_STORAGE_CONTAINER_NAME` below), leave access level as **Private**,
   click **Create**. (The app also creates this automatically on first
   upload if you skip this step — but doing it here lets you confirm the
   storage account works.)

## 4. Deploy the backend → Azure App Service (Web App for Containers)

**If your project lives in a monorepo/portfolio repo** (a subfolder
alongside other unrelated projects) rather than its own dedicated repo, do
this first — it avoids a real gap in Azure's wizard (confirmed against
Microsoft's own docs): the portal's "GitHub Actions" deploy option doesn't
reliably expose a field for building from a subfolder. Instead of hand-
editing the auto-generated workflow YAML (fragile), add one small proxy
file:

- On GitHub, go to your repo's **main page** (not inside any folder) →
  **Add file → Create new file** → name it exactly `Dockerfile` (no
  extension, no folder prefix, right at the repo root) → paste in the
  contents of `for-your-repo-root/Dockerfile` from this zip → commit to
  `main`.
- This is a thin proxy: it just copies from `Azure_RAG_Assistant/backend/`
  internally, so Azure's default "build from repo root" behavior works with
  zero extra configuration. (If your project is in its own dedicated repo
  instead, skip this — the real `backend/Dockerfile` is already at a path
  Azure will find by default.)

Now create the Web App:

1. In the Azure Portal search bar → **"App Services"** → **+ Create** →
   **Web App**.
2. Fill in:
   - **Resource group**: reuse `omnirag-rg` from step 3.
   - **Name**: something unique, e.g. `omnirag-app` — this becomes part of
     your URL: `https://omnirag-app.azurewebsites.net`.
   - **Publish**: select **Docker Container**.
   - **Operating System**: Linux.
   - **Region**: same region as your storage account.
   - **Pricing plan**: click "Change size" / choose plan → pick the **Free
     F1** tier (or the lowest tier your student credit shows as included).
3. On the **Docker** tab (next step in the wizard):
   - **Options**: Single Container.
   - **Image Source**: choose **GitHub Actions** (this is the easiest —
     Azure will auto-build from your repo on every push).
   - Connect your GitHub account, pick your repo, branch `main`.
   - If the wizard shows a Dockerfile/context field, leave it on defaults
     (repo root) — that's exactly what the proxy Dockerfile above is set up
     for.
4. Click **Review + create**, then **Create**. Azure sets up a GitHub
   Actions workflow in your repo automatically and starts the first build —
   this takes a few minutes.
5. Once deployed, go to the App Service's **Configuration** blade (left
   sidebar) → **Application settings** → **+ New application setting**, and
   add each of these one at a time, then click **Save**:
   - `GEMINI_API_KEY`
   - `HF_TOKEN`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `QDRANT_COLLECTION_NAME` = `azure-rag-assistant`
   - `APP_API_KEY` = any random string you make up
   - `AZURE_STORAGE_CONNECTION_STRING` = the connection string from step 3.4
   - `AZURE_STORAGE_CONTAINER_NAME` = `omnirag-documents`
   - `WEBSITES_PORT` = `8000` (tells Azure which port your container listens on)

**If the build fails:** go to your repo's **Actions** tab on GitHub, click
the failed run, and read the error. If it can't find `requirements.txt` or
similar, the proxy `Dockerfile` likely isn't exactly at the repo root —
double check its path is `AI_Portfolio/Dockerfile`, not e.g.
`AI_Portfolio/Azure_RAG_Assistant/Dockerfile`.
6. Give it a minute to restart with the new settings, then open
   `https://<your-app-name>.azurewebsites.net` — you should see the chat UI.
   Confirm the API too: `https://<your-app-name>.azurewebsites.net/health`
   → `{"status":"ok"}`.

From now on, every push to your GitHub repo triggers an automatic redeploy
via the GitHub Actions workflow Azure created for you.

---

## What you can now honestly put on your CV

This is the real thing, so this is fully earned:

> *"Built and deployed a cloud-native agentic RAG system on Microsoft
> Azure — FastAPI backend containerized with Docker, deployed via Azure
> App Service with CI/CD through GitHub Actions, using Azure Blob Storage
> for document archival and Qdrant for vector search."*

You can describe the architecture, defend every piece of it, and point to
a live `.azurewebsites.net` URL if asked.

---

## Optional: local development (Azurite, not real Azure)

`docker-compose.yml` spins up **Azurite** — Microsoft's official free local
emulator for Azure Storage — plus the backend, entirely on your own
machine, no Azure account needed for this part. This is purely for faster
local iteration; it's not required for the real Azure deployment above and
uses a fixed, publicly-documented Azurite development key (never a real
credential).

```bash
docker-compose up --build
# open http://localhost:8000
```

### Running the tests

```bash
cd backend
pip install -r requirements.txt
pytest -v
```

These don't need real API keys or Azure — they only exercise the safe
calculator, the `/health` route, and the root HTML page.

## What changed vs. a typical "eval() calculator" tutorial version

If you've seen versions of this project floating around that use Python's
`eval()` for the calculator tool: **don't use those**. `eval()` on
LLM-controlled input is a remote code execution vulnerability. This project
uses `backend/safe_math.py`, an AST-based evaluator that can't execute
arbitrary code. See `backend/tests/test_safe_math.py` for tests confirming
exploit payloads are rejected.

This project also uses `langchain.agents.create_agent` (the current,
supported LangChain agent API) rather than the deprecated
`initialize_agent`/`AgentExecutor` pattern, and pins dependency versions
verified to install together without conflicts as of this writing — check
`backend/requirements.txt` if you hit version drift later, since the LLM
ecosystem moves fast.

## Project structure

```
Azure_RAG_Assistant/
├── .env.example
├── .gitignore
├── docker-compose.yml           # optional, local dev only (uses Azurite)
├── README.md
├── for-your-repo-root/          # read this if adding to a monorepo (see Step 4)
│   ├── Dockerfile
│   └── README.md
├── .github/workflows/ci.yml     # runs pytest on every push
└── backend/
    ├── Dockerfile
    ├── requirements.txt
    ├── config.py                # centralized settings (pydantic-settings)
    ├── azure_storage.py         # real Azure Blob Storage, fails gracefully if unset
    ├── safe_math.py             # AST-based safe calculator (no eval)
    ├── text_processing.py       # PDF/image/text extraction + chunking + upsert
    ├── agent.py                 # create_agent-based RAG agent
    ├── main.py                  # FastAPI app: "/", /health, /chat, /upload
    ├── static/
    │   └── index.html           # the entire frontend - vanilla HTML/JS, no framework
    └── tests/
        ├── test_safe_math.py
        └── test_health.py
```
