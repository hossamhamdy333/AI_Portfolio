import os

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_tools(config):
    gemini_key = os.environ.get("GEMINI_API_KEY")
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    if not gemini_key or not qdrant_key:
        raise RuntimeError("Set GEMINI_API_KEY and QDRANT_API_KEY first.")

    from crewai import LLM
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    from search_tool import build_search_tool

    qdrant_url = os.environ.get("QDRANT_URL", config["qdrant"]["url"])
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    prefix = config["qdrant"]["collection_prefix"]
    names = {c.name for c in client.get_collections().collections}
    missing = [f"{prefix}_{d}" for d in config["domains"] if f"{prefix}_{d}" not in names]
    if missing:
        raise RuntimeError(f"Missing Qdrant collections: {missing}")

    embed_model = SentenceTransformer(config["embedding"]["model_name"])
    search_tool = build_search_tool(
        client,
        embed_model,
        domains=config["domains"],
        collection_prefix=prefix,
        top_k=config["search"]["top_k"],
    )
    llm = LLM(
        model=config["llm"]["model"],
        api_key=gemini_key,
        temperature=config["llm"]["temperature"],
    )
    return llm, search_tool
