import re
import numpy as np


def split_sentences(text: str):
    sentences = re.split(r"(?<=[.!\u061f])\s+", text.strip())
    return [s for s in sentences if s]


def chunk_fixed(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def chunk_sentence(text: str, chunk_size: int):
    sentences = split_sentences(text)
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_semantic(text: str, embedder, similarity_threshold: float = 0.2):
    """Semantic chunking: embeds all sentences in one batched call, then
    splits where similarity between consecutive sentences drops below
    the threshold. Batching avoids one encode() call per sentence.
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    embeddings = embedder.encode(sentences, batch_size=64, show_progress_bar=False)
    chunks = []
    current = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i], embeddings[i-1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]) + 1e-8
        )
        if sim < similarity_threshold:
            chunks.append(" ".join(current))
            current = []
        current.append(sentences[i])

    if current:
        chunks.append(" ".join(current))
    return chunks
