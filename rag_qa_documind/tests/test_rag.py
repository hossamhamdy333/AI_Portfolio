import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingest import chunk_text  # noqa: E402


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_single_short_paragraph():
    text = "This is a short paragraph."
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=120)
    assert len(chunks) == 1
    assert "short paragraph" in chunks[0]


def test_chunk_text_respects_chunk_size_roughly():
    text = "\n\n".join(["Paragraph number {}. ".format(i) * 20 for i in range(10)])
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    # allow slack for the overlap prefix added onto each chunk
    for c in chunks:
        assert len(c) <= 200 + 40 + 5


def test_chunk_text_overlap_present():
    text = "\n\n".join(["Paragraph {}. ".format(i) * 30 for i in range(6)])
    chunks = chunk_text(text, chunk_size=150, chunk_overlap=50)
    if len(chunks) > 1:
        # the tail of a chunk should reappear at the start of the next
        overlap_sample = chunks[0][-20:]
        assert overlap_sample[:10] in chunks[1]
