"""
Proves the observability layer actually catches a quality regression, not
just logs happy-path traffic.

Runs the same question set through the agent twice - once with the normal
retriever (top_k=3) and once artificially degraded (top_k=1) - so the two
runs show up as separate, comparable traces in Phoenix (project:
azure-rag-assistant): retrieval latency, chunk count, and token usage per
step are all visible per-trace, and the "degraded" run's answers are
visibly thinner because they're grounded in less context.

This requires a real user account with documents already uploaded (via
/upload, while logged in) so there's something for the retriever to find.

Run:
    ENABLE_OBSERVABILITY=true python scripts/regression_demo.py --user-id 1
Then open http://localhost:6006 (or your Phoenix Cloud project) and
compare the "top_k=3" and "top_k=1" traces side by side.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from observability import setup_observability
from agent import build_agent, run_agent

QUESTIONS = [
    "What is the refund policy?",
    "Summarize the key obligations in the uploaded contract.",
    "Who is responsible for late deliveries?",
    "What are the termination conditions?",
]


def run_pass(label, user_id, top_k):
    print(f"\n--- {label} (top_k={top_k}) ---")
    agent = build_agent(user_id=user_id, top_k=top_k)
    for question in QUESTIONS:
        start = time.perf_counter()
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        elapsed = time.perf_counter() - start
        answer = result["messages"][-1].content
        print(f"[{elapsed:.2f}s] Q: {question}\n         A: {str(answer)[:200]}")


def main(user_id):
    if os.environ.get("ENABLE_OBSERVABILITY", "false").lower() == "true":
        setup_observability()
    else:
        print("ENABLE_OBSERVABILITY is not set - running without tracing. "
              "Set it to true to actually send these two runs to Phoenix.")

    run_pass("normal", user_id, top_k=3)
    run_pass("degraded", user_id, top_k=1)

    print("\nDone. If observability was enabled, open Phoenix and compare "
          "the two runs - the degraded pass should show fewer retrieved "
          "chunks and thinner, less-grounded answers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, required=True, help="A real user ID that has documents uploaded")
    args = parser.parse_args()
    main(args.user_id)
