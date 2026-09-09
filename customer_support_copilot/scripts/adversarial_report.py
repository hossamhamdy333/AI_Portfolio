"""
Runs the adversarial test set through the guardrails layer and reports the
catch rate honestly - including what it misses.

Run:  python scripts_adversarial_report.py
"""

import json
import sys
sys.path.insert(0, "src")
import guardrails


def main():
    with open("src/adversarial_prompts.json") as f:
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

    total_correct = sum(c for c, _ in by_category.values())
    total = sum(t for _, t in by_category.values())

    print(f"Overall: {total_correct}/{total} correct ({100 * total_correct / total:.0f}%)\n")
    for cat, (correct, total_cat) in by_category.items():
        print(f"{cat:<18} {correct}/{total_cat}")

    if misses:
        print("\nMisses:")
        for prompt, cat, got, want in misses:
            print(f"  [{cat}] got={got} want={want} :: {prompt}")


if __name__ == "__main__":
    main()
