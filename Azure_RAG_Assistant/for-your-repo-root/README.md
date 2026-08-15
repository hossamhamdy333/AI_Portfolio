# This file goes somewhere different than the rest of this zip

The `Dockerfile` in this folder is **not** meant to stay inside
`Azure_RAG_Assistant/`. It needs to end up at the **root** of your `AI_Portfolio`
GitHub repo (i.e. `AI_Portfolio/Dockerfile`, as a sibling of
`Azure_RAG_Assistant/`, `Credit_Fraud_Detection/`, etc.) - NOT inside this
folder, and not inside `Azure_RAG_Assistant/backend/` either.

Why: Azure App Service's GitHub Actions deploy wizard defaults to building
from a Dockerfile at the repo root. Since `AI_Portfolio` is a monorepo with
many unrelated project folders, this thin proxy Dockerfile at the repo root
just points into `Azure_RAG_Assistant/backend/` for everything it needs, so the
default wizard behavior works without editing any auto-generated workflow
YAML.

Easiest way to add it: on GitHub, go to your `AI_Portfolio` repo's main
page (not inside any folder), click **Add file → Create new file**, name it
exactly `Dockerfile` (no extension, no folder prefix), paste in the content
of the `Dockerfile` next to this README, and commit directly to `main`.

See the main `README.md` (in the `Azure_RAG_Assistant` folder) for the full
deployment walkthrough.
