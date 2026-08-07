_DOCUMENTS = [
    {"id": 1, "title": "Gemini API quickstart", "text": "How to call the Gemini API for text generation."},
    {"id": 2, "title": "Function calling guide", "text": "How the model requests tool calls and how to respond."},
    {"id": 3, "title": "Structured outputs", "text": "Using Pydantic schemas to validate model responses."},
]


def get_current_weather(city: str) -> dict:
    """Stub — swap in a real weather API if you want; not required here."""
    return {"city": city, "temp_c": 24, "condition": "clear"}


def search_documents(query: str) -> list[dict]:
    """Naive keyword search over the in-memory corpus.

    Same interface a real vector search would expose (query in, ranked
    docs out) — only the implementation underneath would change.
    """
    query_lower = query.lower()
    matches = [doc for doc in _DOCUMENTS if query_lower in doc["text"].lower() or query_lower in doc["title"].lower()]
    return matches or [{"id": None, "title": "No match", "text": "No matching documents found."}]


# Maps the name Gemini returns in a function call to the actual Python callable.
TOOL_REGISTRY = {
    "get_current_weather": get_current_weather,
    "search_documents": search_documents,
}


def run_tool(tool_name: str, arguments: dict):
    """Look up and execute the tool the model asked for.

    Raising on an unknown tool name is intentional — silently ignoring it
    would hide a real bug (model hallucinating a tool, or a typo in the
    tool schema we sent it).
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool requested: {tool_name}")
    return TOOL_REGISTRY[tool_name](**arguments)
