import json
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "outputs" / "audit" / "audit_log.jsonl"


def load_runs(path=LOG):
    runs = {}
    if not Path(path).exists():
        return runs
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                runs.setdefault(row["run_id"], []).append(row)
    return runs


def format_run(rows, width=110):
    lines = []
    for r in rows:
        extra = {k: v for k, v in r.items() if k not in ("run_id", "time", "step")}
        text = " | ".join(f"{k}={str(v)[:width]}" for k, v in extra.items())
        lines.append(f"{r['time']}  {r['step']:<19} {text}")
    return "\n".join(lines)


def main():
    runs = load_runs()
    if not runs:
        sys.exit("No audit log found.")

    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        for run_id, rows in runs.items():
            final = next((r["status"] for r in rows if r["step"] == "final"), "unfinished")
            print(run_id, rows[0]["time"], final, rows[0].get("question", ""))
        return

    run_id = sys.argv[1] if len(sys.argv) > 1 else list(runs)[-1]
    if run_id not in runs:
        sys.exit(f"Run {run_id} not found. Use --list to see run ids.")
    print(format_run(runs[run_id]))


if __name__ == "__main__":
    main()
