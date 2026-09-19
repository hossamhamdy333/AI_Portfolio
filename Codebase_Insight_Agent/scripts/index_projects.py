"""Index only the projects you name, without rebuilding the rest.

Run from the project folder:
    python scripts/index_projects.py Codebase_Insight_Agent portfolio_overview
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()  # has to run before config is imported, config reads the env once

import config
import portfolio


def main():
    names = sys.argv[1:]
    if not names:
        print("Usage: python scripts/index_projects.py <project_name> [<project_name> ...]")
        sys.exit(1)

    unknown = [name for name in names if name not in config.PROJECTS]
    if unknown:
        print(f"Not in config.PROJECTS: {', '.join(unknown)}")
        sys.exit(1)

    if not config.QDRANT_URL:
        print("QDRANT_URL is empty. Fill in .env first, otherwise the index only lives in memory.")
        sys.exit(1)

    client = portfolio.get_qdrant_client()
    for name in names:
        print(f"Indexing {name}...")
        portfolio.build_index(name, client)
    print("Done.")


if __name__ == "__main__":
    main()
