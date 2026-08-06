"""
Document ingestion pipeline:
  load file -> extract text -> chunk text -> embed chunks -> store in vector DB
"""
import os
import uuid
from typing import List

from pypdf import PdfReader

from app.config import settings
from app.vectorstore import get_collection


def load_text(file_path: str) -> str:
    """Extract raw text from a .txt, .md, or .pdf file."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        # "layout" mode preserves visual spacing far more faithfully than the
        # default "plain" mode -- some PDFs (common with LaTeX-generated
        # academic/textbook PDFs) encode text without explicit space
        # characters between words, relying on visual positioning instead.
        # Plain mode loses that positioning and runs words together
        # ("Sofarwehavegivenafairly..."); layout mode reconstructs spacing
        # from each character's actual position on the page.
        return "\n".join(
            page.extract_text(extraction_mode="layout") or "" for page in reader.pages
        )

    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[str]:
    """
    Split text into overlapping chunks by character count, breaking on
    paragraph/sentence boundaries where possible so chunks stay coherent.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # paragraph itself is longer than chunk_size -> hard split it
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - chunk_overlap):
                    chunks.append(para[i:i + chunk_size])
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    # add overlap between consecutive chunks for better retrieval context
    overlapped = []
    for i, c in enumerate(chunks):
        if i == 0:
            overlapped.append(c)
        else:
            tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append(f"{tail} {c}".strip())

    return overlapped


def ingest_file(file_path: str, source_name: str = None) -> int:
    """
    Full pipeline for one file: load -> chunk -> embed -> upsert into Chroma.
    Returns the number of chunks stored.
    """
    source_name = source_name or os.path.basename(file_path)
    text = load_text(file_path)
    chunks = chunk_text(text)

    if not chunks:
        return 0

    collection = get_collection()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def ingest_directory(directory: str) -> dict:
    """Ingest every supported file in a directory. Returns a summary dict."""
    summary = {}
    for fname in os.listdir(directory):
        if fname.lower().endswith((".txt", ".md", ".pdf")):
            path = os.path.join(directory, fname)
            summary[fname] = ingest_file(path, source_name=fname)
    return summary
