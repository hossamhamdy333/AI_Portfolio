"""
CLI helper to ingest all documents in a directory.

Usage:
    python scripts/run_ingest.py data/sample_docs
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingest import ingest_directory  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_ingest.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Not a directory: {directory}")
        sys.exit(1)

    summary = ingest_directory(directory)
    if not summary:
        print("No supported files (.txt, .md, .pdf) found.")
        return

    print("Ingestion summary:")
    for fname, n_chunks in summary.items():
        print(f"  {fname}: {n_chunks} chunks indexed")


if __name__ == "__main__":
    main()
