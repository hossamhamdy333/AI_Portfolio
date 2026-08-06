"""Tools available to agents — all free."""
import io
import contextlib
from ddgs import DDGS  # duckduckgo_search was renamed to ddgs upstream


def web_search(query: str, max_results: int = 5) -> str:
    """
    DDG search is free/no-key but scraping-based, so it rate-limits under bursts
    of requests (HTTP 403). We don't let that crash the whole agent run — it just
    means the Researcher falls back to whatever's in the vector store for that turn.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Web search unavailable right now ({e.__class__.__name__}). Relying on indexed documents only."
    if not results:
        return "No results found."
    return "\n\n".join(f"- {r['title']}: {r['body']} ({r['href']})" for r in results)


def retrieve(qdrant_client, query: str, k: int = 4) -> str:
    from ingestion.chunking import search
    hits = search(qdrant_client, query, k=k)
    if not hits:
        return "No indexed documents found."
    return "\n\n".join(f"[{h['source']} p.{h['page']}] {h['text']}" for h in hits)


# Demo-grade sandbox: restricted builtins, no file/network access from inside exec.
# Good enough for a personal portfolio project — say so in the README, don't expose
# this endpoint publicly without a real sandbox (e.g. subprocess + resource limits,
# or a container) if it's ever meant to take untrusted input.
ALLOWED_IMPORTS = {"pandas", "numpy", "math", "statistics"}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    # LLM-generated code habitually writes "import pandas as pd" even though
    # pd/np are already injected below — without this, that line crashes with
    # "__import__ not found" since it's excluded from SAFE_BUILTINS.
    if name not in ALLOWED_IMPORTS:
        raise ImportError(f"'{name}' is not an allowed import in this sandbox")
    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    "print": print, "range": range, "len": len, "sum": sum, "min": min, "max": max,
    "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "str": str, "int": int,
    "float": float, "bool": bool, "__import__": _safe_import,
}


def run_python(code_str: str) -> str:
    buf = io.StringIO()
    local_env = {}
    try:
        with contextlib.redirect_stdout(buf):
            import pandas as pd
            import numpy as np
            exec(code_str, {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np}, local_env)
    except Exception as e:
        return f"Error: {e}"
    output = buf.getvalue()
    return output if output else "Code ran with no printed output."
