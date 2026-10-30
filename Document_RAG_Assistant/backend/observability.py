"""
LLM observability via Arize Phoenix (OpenTelemetry-based tracing).

Once enabled, every LangChain call the agent makes - the retriever, the
LLM call, the calculator tool - is automatically traced: latency per
step, token counts, and the retrieved context itself, viewable in the
Phoenix UI.

Not called at import time, so importing this module never requires a
running Phoenix collector: call `setup_observability()` once, at app
startup, only if observability is actually turned on (see main.py).
"""

import os

_initialized = False


def setup_observability():
    """Start sending traces to Phoenix.

    Self-hosted (free, local):
        docker run -p 6006:6006 arizephoenix/phoenix:latest
        export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006

    Or Phoenix Cloud: set PHOENIX_API_KEY instead of the endpoint above.
    """
    global _initialized
    if _initialized:
        return

    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    tracer_provider = register(
        project_name="azure-rag-assistant",
        endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"),
        auto_instrument=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    _initialized = True
