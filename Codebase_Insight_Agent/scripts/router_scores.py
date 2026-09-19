"""Show how the router scores a question against every project.

Run from the project folder:
    python scripts/router_scores.py "What skills does Hossam have?"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

import config
import portfolio


def main():
    if len(sys.argv) != 2:
        print('Usage: python scripts/router_scores.py "your question"')
        sys.exit(1)

    question = sys.argv[1]
    router = portfolio.build_router()

    print(f"Threshold: {config.SIMILARITY_THRESHOLD}\n")
    for name, score in router.scores(question)[:8]:
        print(f"{score:.3f}  {name}")
    print("\nPicked:", ", ".join(router.select(question)))


if __name__ == "__main__":
    main()
