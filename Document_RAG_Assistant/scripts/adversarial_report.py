"""
Runs the adversarial test set through the guardrails layer and reports the
catch rate honestly - including what it misses. This is a report, not a
pytest suite: it isn't meant to always pass, it's meant to be read.

Run:  python scripts/adversarial_report.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import guardrails


def main():
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "backend", "tests", "adversarial_prompts.json"
    )
    with open(data_path) as f:
        cases = json.load(f)

    by_category = {}
    misses = []

    for case in cases:
        result = guardrails.guard_input(case["prompt"])
        got_blocked = result["blocked"]
        want_blocked = case["expect_blocked"]
        correct = got_blocked == want_blocked

        cat = case["category"]
        by_category.setdefault(cat, [0, 0])
        by_category[cat][1] += 1
        if correct:
            by_category[cat][0] += 1
        else:
            misses.append((case["prompt"], cat, got_blocked, want_blocked))

        if case.get("expect_redaction") and result["pii_redactions"] == 0:
            misses.append((case["prompt"], cat, "no redaction", "expected redaction"))

    total_correct = sum(c for c, _ in by_category.values())
    total = sum(t for _, t in by_category.values())

    print(f"Overall: {total_correct}/{total} correct ({100 * total_correct / total:.0f}%)\n")
    print(f"{'category':<18} correct")
    for cat, (correct, total_cat) in by_category.items():
        print(f"{cat:<18} {correct}/{total_cat}")

    if misses:
        print("\nMisses (read these, don't hide them):")
        for prompt, cat, got, want in misses:
            print(f"  [{cat}] got={got} want={want} :: {prompt}")
    else:
        print("\nNo misses.")


if __name__ == "__main__":
    main()
