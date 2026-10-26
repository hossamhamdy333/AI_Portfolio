import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from llama_index.core.selectors import EmbeddingSingleSelector
from llama_index.core.tools import ToolMetadata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from router import DOMAIN_DESCRIPTIONS

TOLERANCE = 0.03


def route_all(questions, domains, embed_model):
    selector = EmbeddingSingleSelector.from_defaults(embed_model=embed_model)
    choices = [ToolMetadata(name=d, description=DOMAIN_DESCRIPTIONS[d]) for d in domains]
    routed = []
    for question in questions["question"]:
        try:
            result = selector.select(choices, question)
            routed.append(domains[result.selections[0].index])
        except Exception:
            routed.append(None)
    return routed


def score(questions, routed, domains):
    right = pd.Series([a == b for a, b in zip(questions["domain"], routed)])
    per_domain = {d: float(right[questions["domain"].values == d].mean()) for d in domains}
    return {
        "routing_accuracy": round(float(right.mean()), 4),
        "worst_domain_accuracy": round(min(per_domain.values()), 4),
    }


def failed_metrics(scores, baseline, tolerance=TOLERANCE):
    return [name for name, base in baseline.items() if scores[name] < base - tolerance]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
    domains = cfg["domains"]
    questions = pd.read_parquet(HERE / "questions.parquet")
    print("questions:", len(questions))

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embed_model = HuggingFaceEmbedding(model_name=cfg["embedding"]["model_name"])

    routed = route_all(questions, domains, embed_model)
    print("selector failures:", sum(r is None for r in routed))
    scores = score(questions, routed, domains)
    print("scores:", json.dumps(scores))

    baseline_path = HERE / "baseline.json"
    if args.write_baseline:
        baseline_path.write_text(json.dumps(scores, indent=2))
        print("baseline written")
        return
    if not baseline_path.exists():
        print("No eval/baseline.json yet. Copy the scores line above into it.")
        sys.exit(1)

    baseline = json.loads(baseline_path.read_text())
    for name, base in baseline.items():
        print(f"{name}: now {scores[name]} | baseline {base}")
    failed = failed_metrics(scores, baseline)
    if failed:
        print("GATE FAILED, dropped more than", TOLERANCE, "on:", ", ".join(failed))
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
