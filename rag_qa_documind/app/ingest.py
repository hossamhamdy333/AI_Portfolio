"""
Document ingestion pipeline:
  load file -> extract text -> chunk text -> embed chunks -> store in vector DB
"""
import os
<<<<<<< HEAD
import uuid
from typing import List

=======
import re
import uuid
from typing import List

import wordninja
>>>>>>> 1daa2d74e09f7db542620d4ab4861f9cf5e0dc25
from pypdf import PdfReader

from app.config import settings
from app.vectorstore import get_collection


<<<<<<< HEAD
=======
def _fix_concatenated_words(text: str, min_len: int = 20) -> str:
    """
    Some PDFs (common with certain LaTeX-generated academic/textbook PDFs)
    extract with no spaces between words at all -- pypdf's "layout" mode
    fixes this for most PDFs by reconstructing spacing from character
    position, but some PDFs' internal encoding has no positional gap to
    detect. As a robust fallback that works regardless of the root cause,
    any individual "word" that's suspiciously long (a strong sign several
    real words got fused together, e.g. "Sofarwehavegivenafairly...") gets
    run through dictionary-based word segmentation. Normal-length words are
    left completely untouched, so ordinary text, punctuation, and
    capitalization are unaffected.
    """
    def fix_token(match):
        token = match.group(0)
        core = token.rstrip(".,:;!?)")
        trailing = token[len(core):]
        if len(core) >= min_len:
            return " ".join(wordninja.split(core)) + trailing
        return token

    return re.sub(r"\S+", fix_token, text)


>>>>>>> 1daa2d74e09f7db542620d4ab4861f9cf5e0dc25
def load_text(file_path: str) -> str:
    """Extract raw text from a .txt, .md, or .pdf file."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
<<<<<<< HEAD
        return "\n".join(page.extract_text() or "" for page in reader.pages)
=======
        # "layout" mode preserves visual spacing far more faithfully than the
        # default "plain" mode for most PDFs with missing word spaces.
        text = "\n".join(
            page.extract_text(extraction_mode="layout") or "" for page in reader.pages
        )
        # Belt-and-suspenders: fix any words that are still fused together
        # even after layout-mode extraction.
        return _fix_concatenated_words(text)
>>>>>>> 1daa2d74e09f7db542620d4ab4861f9cf5e0dc25

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
<<<<<<< HEAD
            # paragraph itself is longer than chunk_size -> hard split it
=======
>>>>>>> 1daa2d74e09f7db542620d4ab4861f9cf5e0dc25
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - chunk_overlap):
                    chunks.append(para[i:i + chunk_size])
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

<<<<<<< HEAD
    # add overlap between consecutive chunks for better retrieval context
=======
>>>>>>> 1daa2d74e09f7db542620d4ab4861f9cf5e0dc25
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
