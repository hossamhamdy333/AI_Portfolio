import json
import os

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
