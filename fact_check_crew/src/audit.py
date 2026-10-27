import json
import time
import uuid
from pathlib import Path


class AuditLog:
    def __init__(self, path="outputs/audit/audit_log.jsonl", run_id=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:8]

    def log(self, step, **data):
        row = {
            "run_id": self.run_id,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            **data,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def read(self):
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
