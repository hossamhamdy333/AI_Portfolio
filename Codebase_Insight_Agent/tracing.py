import json
import math
import os
from collections import deque

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

TRACE_FILE = os.environ.get("TRACE_FILE", "traces.jsonl")


class JsonlExporter(SpanExporter):
    def __init__(self, path):
        self.path = path

    def export(self, spans):
        lines = []
        for s in spans:
            parent = s.parent.span_id if s.parent else None
            lines.append(json.dumps({
                "trace_id": format(s.context.trace_id, "032x"),
                "span_id": format(s.context.span_id, "016x"),
                "parent_id": format(parent, "016x") if parent else None,
                "name": s.name,
                "start": s.start_time / 1e9,
                "duration_s": round((s.end_time - s.start_time) / 1e9, 3),
                "error": s.status.status_code.name == "ERROR",
                "attributes": dict(s.attributes or {}),
            }))
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError:
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


_provider = TracerProvider(resource=Resource.create({"service.name": "codebase-insight-agent"}))
_provider.add_span_processor(SimpleSpanProcessor(JsonlExporter(TRACE_FILE)))
_tracer = _provider.get_tracer("codebase-insight")


def span(name, **attributes):
    return _tracer.start_as_current_span(name, attributes=attributes)


def token_counts(response):
    usage = getattr(response, "usage_metadata", None) or {}
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


def read_spans(path, max_lines=20000):
    # only the tail of the file, so a long-running server doesn't read it all
    try:
        with open(path, encoding="utf-8") as f:
            lines = deque(f, maxlen=max_lines)
    except OSError:
        return []
    spans = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            spans.append(json.loads(line))
        except ValueError:
            continue
    return spans


def _percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[index]


def _ask_row(span):
    attrs = span.get("attributes", {})
    return {
        "start": span["start"],
        "duration_s": span["duration_s"],
        "error": span["error"],
        "retries": attrs.get("retries") or 0,
        "projects": attrs.get("projects") or "",
    }


def summarize(spans, input_per_m=0.0, output_per_m=0.0, recent=40):
    by_name = {}
    for s in spans:
        by_name.setdefault(s["name"], []).append(s)

    order = ["ask", "plan", "retrieve", "lookup", "draft", "critique"]
    names = [n for n in order if n in by_name] + sorted(n for n in by_name if n not in order)
    table = []
    for name in names:
        group = by_name[name]
        times = [g["duration_s"] for g in group]
        table.append({
            "name": name,
            "count": len(group),
            "avg_s": round(sum(times) / len(times), 3),
            "p95_s": round(_percentile(times, 95), 3),
            "max_s": round(max(times), 3),
            "errors": sum(1 for g in group if g["error"]),
        })

    asks = sorted(by_name.get("ask", []), key=lambda a: a["start"])
    times = [a["duration_s"] for a in asks]
    failed = sum(1 for a in asks if a["error"])
    retried = sum(1 for a in asks if (a.get("attributes", {}).get("retries") or 0) > 0)

    input_tokens = sum(s.get("attributes", {}).get("input_tokens", 0) for s in spans)
    output_tokens = sum(s.get("attributes", {}).get("output_tokens", 0) for s in spans)
    cost = input_tokens / 1e6 * input_per_m + output_tokens / 1e6 * output_per_m

    return {
        "questions": len(asks),
        "failed": failed,
        "error_rate": round(failed / len(asks), 4) if asks else 0.0,
        "retried": retried,
        "avg_s": round(sum(times) / len(times), 3) if times else 0.0,
        "p95_s": round(_percentile(times, 95), 3),
        "max_s": round(max(times), 3) if times else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "est_cost_usd": round(cost, 6),
        "cost_per_question_usd": round(cost / len(asks), 6) if asks else 0.0,
        "price_per_m": {"input": input_per_m, "output": output_per_m},
        "spans": table,
        "recent": [_ask_row(a) for a in asks[-recent:]],
        "slowest": [_ask_row(a) for a in sorted(asks, key=lambda a: -a["duration_s"])[:5]],
    }
