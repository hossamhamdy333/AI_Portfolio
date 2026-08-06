"""ParentDocumentRetriever over the same XLSum Arabic corpus impl_vanilla uses.

This is the concrete "does the abstraction earn its complexity" test: vanilla
embeds and searches flat fixed-size chunks. Here, small child chunks get
embedded and searched, but the *parent* chunk (more surrounding context) is
what actually gets passed to the LLM -- something vanilla's flat-chunk
design can't do without restructuring its whole pipeline.

Loads data/processed/xlsum_arabic_clean.parquet (produced by impl_vanilla's
01_eda.ipynb / data_utils.py -- pull it via DVC same as vanilla does, don't
recreate the cleaning step here).
"""

from pathlib import Path

import pandas as pd
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_articles_as_documents(parquet_path):
    df = pd.read_parquet(parquet_path)
    return [
        Document(
            page_content=row["article"],
            metadata={"article_id": row["id"], "title": row.get("title", "")},
        )
        for _, row in df.iterrows()
    ]


def build_parent_document_retriever(
    documents,
    embedding_model_name,
    parent_chunk_size=1200,
    child_chunk_size=300,
    child_chunk_overlap=32,
    persist_directory=None,
    batch_size=200,
    device=None,
):
    """Mirrors impl_vanilla's chunk_size=256/overlap=32 choice for the child
    splitter (the layer that actually gets embedded and searched), so the
    retrieval-quality comparison in COMPARISON.md isn't confounded by a
    different chunk size on top of the different architecture.

    persist_directory: if set, Chroma writes to disk instead of holding the
    whole vector index in RAM -- avoids the OOM crash that a single
    in-memory add_documents() call over 15K+ articles can trigger on Colab.
    batch_size: documents are added in batches with a progress bar so a
    slow-but-working run doesn't look identical to a frozen/crashed one.
    device: 'cuda' or 'cpu'. Auto-detects GPU if not specified -- CPU
    embedding of 100K+ chunks can take hours, GPU cuts that dramatically.
    """
    import torch
    from tqdm.auto import tqdm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding on device: {device}")

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 64},
    )
    chroma_kwargs = {"collection_name": "rag_documents_langchain", "embedding_function": embeddings}
    if persist_directory:
        chroma_kwargs["persist_directory"] = persist_directory
    vectorstore = Chroma(**chroma_kwargs)
    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches"):
        retriever.add_documents(documents[i:i + batch_size])

    return retriever


def build_retriever_from_config(config, data_dir):
    documents = load_articles_as_documents(Path(data_dir) / "xlsum_arabic_clean.parquet")
    return build_parent_document_retriever(
        documents,
        embedding_model_name=config["retrieval"]["model_name"],
        child_chunk_size=config["chunking"]["chunk_size"],
        child_chunk_overlap=config["chunking"]["overlap"],
    )
