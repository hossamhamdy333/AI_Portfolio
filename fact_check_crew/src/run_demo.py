import getpass
import os
import sys
from pathlib import Path

import yaml

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

ROOT = Path(__file__).resolve().parent.parent


def get_key(name):
    value = os.environ.get(name)
    if not value:
        value = getpass.getpass(f"{name}: ")
        os.environ[name] = value
    return value


def main():
    question = " ".join(sys.argv[1:]).strip() or input("Question: ").strip()

    with open(ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    gemini_key = get_key("GEMINI_API_KEY")
    qdrant_key = get_key("QDRANT_API_KEY")

    from crewai import LLM
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    from approval import cli_approval
    from audit import AuditLog
    from search_tool import build_search_tool
    from supervisor import run_supervised

    qdrant_url = os.environ.get("QDRANT_URL", config["qdrant"]["url"])
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    prefix = config["qdrant"]["collection_prefix"]
    try:
        names = {c.name for c in qdrant_client.get_collections().collections}
    except Exception as e:
        sys.exit(f"Qdrant check failed: {e}\nCheck QDRANT_API_KEY and the url in configs/config.yaml.")
    missing = [f"{prefix}_{d}" for d in config["domains"] if f"{prefix}_{d}" not in names]
    if missing:
        sys.exit(f"Missing Qdrant collections: {missing}")
    embed_model = SentenceTransformer(config["embedding"]["model_name"])
    search_tool = build_search_tool(
        qdrant_client,
        embed_model,
        domains=config["domains"],
        collection_prefix=config["qdrant"]["collection_prefix"],
        top_k=config["search"]["top_k"],
    )
    llm = LLM(
        model=config["llm"]["model"],
        api_key=gemini_key,
        temperature=config["llm"]["temperature"],
    )

    audit = AuditLog(ROOT / "outputs" / "audit" / "audit_log.jsonl")
    result = run_supervised(
        llm,
        search_tool,
        question,
        audit,
        cli_approval,
        max_revisions=config["revision"]["max_revisions"],
    )

    print("\nStatus:", result["status"])
    print("Revisions used:", result["n_revisions"])
    if result["answer"]:
        print("\nFinal answer:\n" + result["answer"])
    else:
        print("No answer released.")
    print("\nRun id:", result["run_id"])
    print("Audit log:", audit.path)


if __name__ == "__main__":
    main()
