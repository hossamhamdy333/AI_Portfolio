"""
Streamlit demo for the semantic search project.

Deploy target: Streamlit Community Cloud (share.streamlit.io) - free,
no credit card, just a GitHub login. Point it at this repo, main file
`streamlit_app.py`, and set QDRANT_URL / QDRANT_API_KEY under
Settings -> Secrets (see .env.example for the exact names/format).

This queries the SAME persistent Qdrant Cloud collection that
scripts/build_index.py populates and src/serve.py serves - there's no
separate local index here.
"""
import sys
from pathlib import Path

import streamlit as st
import yaml
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from retrieval import load_model, encode_query, search_qdrant  # noqa: E402

st.set_page_config(page_title="ArXiv Semantic Search", page_icon="🔎", layout="centered")


@st.cache_resource
def get_config():
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_clients(_config):
    from sentence_transformers import CrossEncoder

    qdrant_url = st.secrets.get("QDRANT_URL", None) or st.session_state.get("QDRANT_URL")
    qdrant_key = st.secrets.get("QDRANT_API_KEY", None)

    if not qdrant_url:
        st.error(
            "QDRANT_URL is not set. If you're the app owner, add QDRANT_URL and "
            "QDRANT_API_KEY under Settings -> Secrets in Streamlit Community Cloud."
        )
        st.stop()

    model    = load_model(_config["retrieval"]["model_name"])
    reranker = CrossEncoder(_config["reranker"]["model_name"])
    client   = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    return model, reranker, client


config = get_config()
model, reranker, client = get_clients(config)

st.title("🔎 Semantic Search over ArXiv ML Papers")
st.caption(
    f"Bi-encoder: `{config['retrieval']['model_name']}` -> "
    f"Cross-encoder rerank: `{config['reranker']['model_name']}`"
)

query = st.text_input("Search query", placeholder="e.g. contrastive learning for sentence embeddings")

if query:
    with st.spinner("Searching..."):
        query_vec = encode_query(model, query)[0]
        hits = search_qdrant(
            client,
            config["qdrant"]["collection_name"],
            query_vec,
            top_k=config["retrieval"]["top_k"],
        )

        if not hits:
            st.warning("No results. Has the index been built? See scripts/build_index.py.")
        else:
            pairs  = [[query, h["abstract"]] for h in hits]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
            rerank_k = config["retrieval"]["rerank_top_k"]

            for score, hit in ranked[:rerank_k]:
                st.subheader(hit["title"])
                st.write(hit["abstract"])
                st.caption(f"relevance score: {score:.3f}")
                st.divider()
