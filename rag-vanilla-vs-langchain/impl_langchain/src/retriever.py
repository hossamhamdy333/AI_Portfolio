"""ParentDocumentRetriever over the same XLSum Arabic corpus impl_vanilla uses.

This is the concrete "does the abstraction earn its complexity" test: vanilla
embeds and searches flat fixed-size chunks. Here, small child chunks get
embedded and searched, but the *parent* chunk (more surrounding context) is
what actually gets passed to the LLM -- something vanilla's flat-chunk
design can't do without restructuring its whole pipeline.

Loads data/xlsum_arabic_clean.parquet, a copy of impl_vanilla's cleaned
corpus (produced by impl_vanilla's 01_eda.ipynb / data_utils.py -- pulled
via DVC and copied over, not recreated here).

Persistence: both stores ParentDocumentRetriever needs are now persisted to
disk under persist_directory -- the child-chunk vectors in Chroma (as
before), and the parent documents in a LocalFileStore (langchain's
InMemoryStore, which this used to use, only lives in process memory and is
never written to disk despite `dvc add data/chroma_db` suggesting the whole
retriever was saved). That gap meant every notebook that needed the
retriever had to call build_parent_document_retriever() again from zero --
and since that function always called add_documents() unconditionally, each
rebuild against the same persist_directory silently duplicated the child
chunks already sitting in Chroma. build_parent_document_retriever() now
only embeds+adds when the vectorstore is actually empty; otherwise use
load_parent_document_retriever() to reconnect to what's already on disk
with no re-embedding at all.
"""

from pathlib import Path

import pandas as pd
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, create_kv_docstore
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


def _build_stores(embedding_model_name, persist_directory, device):
    """Shared by build_ and load_ -- both need the same Chroma collection
    and the same on-disk docstore, just with different add_documents()
    behavior on top.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 64},
    )
    vectorstore = Chroma(
        collection_name="rag_documents_langchain",
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    docstore = create_kv_docstore(LocalFileStore(str(Path(persist_directory) / "docstore")))
    return vectorstore, docstore


def build_parent_document_retriever(
    documents,
    embedding_model_name,
    parent_chunk_size=1200,
    child_chunk_size=300,
    child_chunk_overlap=32,
    persist_directory=None,
    batch_size=200,
    device=None,
    force_rebuild=False,
):
    """Mirrors impl_vanilla's chunk_size=256/overlap=32 choice for the child
    splitter (the layer that actually gets embedded and searched), so the
    retrieval-quality comparison in COMPARISON.md isn't confounded by a
    different chunk size on top of the different architecture.

    persist_directory: required now -- both the child-chunk vectors (Chroma)
    and the parent documents (LocalFileStore) are written here, so the
    retriever survives across notebook runs without re-embedding.
    force_rebuild: if the vectorstore already has vectors in it, this
    function is a no-op unless force_rebuild=True -- prevents silently
    duplicating chunks on a second run. Use load_parent_document_retriever()
    for the normal "just give me the existing retriever" case instead.
    batch_size: documents are added in batches with a progress bar so a
    slow-but-working run doesn't look identical to a frozen/crashed one.
    device: 'cuda' or 'cpu'. Auto-detects GPU if not specified -- CPU
    embedding of 100K+ chunks can take hours, GPU cuts that dramatically.
    """
    import torch
    from tqdm.auto import tqdm

    if persist_directory is None:
        raise ValueError("persist_directory is required -- both stores need somewhere to live on disk.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding on device: {device}")

    vectorstore, docstore = _build_stores(embedding_model_name, persist_directory, device)

    existing_count = vectorstore._collection.count()
    if existing_count > 0 and not force_rebuild:
        print(f"Vectorstore already has {existing_count} chunks at {persist_directory} -- "
              f"skipping re-embedding. Pass force_rebuild=True to override.")
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=0)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap)
        return ParentDocumentRetriever(
            vectorstore=vectorstore, docstore=docstore,
            child_splitter=child_splitter, parent_splitter=parent_splitter,
        )

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches"):
        retriever.add_documents(documents[i:i + batch_size])

    return retriever


def load_parent_document_retriever(
    embedding_model_name,
    persist_directory,
    parent_chunk_size=1200,
    child_chunk_size=300,
    child_chunk_overlap=32,
    device=None,
):
    """Reconnect to a retriever already built by build_parent_document_retriever()
    -- no re-embedding, no risk of duplicating chunks. This is what every
    notebook downstream of the one that first builds the retriever should
    call instead of build_parent_document_retriever() again.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    vectorstore, docstore = _build_stores(embedding_model_name, persist_directory, device)
    existing_count = vectorstore._collection.count()
    if existing_count == 0:
        raise RuntimeError(
            f"No existing vectors found at {persist_directory} -- "
            "run build_parent_document_retriever() first."
        )
    print(f"Loaded existing retriever: {existing_count} child chunks at {persist_directory}")

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap)
    return ParentDocumentRetriever(
        vectorstore=vectorstore, docstore=docstore,
        child_splitter=child_splitter, parent_splitter=parent_splitter,
    )


def build_retriever_from_config(config, data_dir, persist_directory):
    documents = load_articles_as_documents(Path(data_dir) / "xlsum_arabic_clean.parquet")
    return build_parent_document_retriever(
        documents,
        embedding_model_name=config["retrieval"]["model_name"],
        child_chunk_size=config["chunking"]["chunk_size"],
        child_chunk_overlap=config["chunking"]["overlap"],
        persist_directory=persist_directory,
    )
