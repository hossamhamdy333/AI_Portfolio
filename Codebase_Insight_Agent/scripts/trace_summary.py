import json
import os
from collections import defaultdict

path = os.environ.get("TRACE_FILE", "traces.jsonl")
spans = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

by_name = defaultdict(list)
errors = defaultdict(int)
for s in spans:
    by_name[s["name"]].append(s["duration_s"])
    errors[s["name"]] += s["error"]

print(f"{'span':10} {'count':>5} {'avg s':>8} {'max s':>8} {'errors':>6}")
for name, times in by_name.items():
    print(f"{name:10} {len(times):>5} {sum(times) / len(times):>8.2f} {max(times):>8.2f} {errors[name]:>6}")

print()
print("slowest questions:")
asks = sorted((s for s in spans if s["name"] == "ask"), key=lambda s: -s["duration_s"])[:5]
for a in asks:
    a_attr = a["attributes"]
    print(f"  {a['duration_s']:6.1f}s  retries={a_attr.get('retries')}  projects={a_attr.get('projects')}")
