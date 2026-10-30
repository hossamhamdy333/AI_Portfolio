import re

_REF = re.compile(r"\[(\d{1,2})\]")
_GROUP = re.compile(r"\[(\d{1,2}(?:\s*,\s*\d{1,2})+)\]")
MAX_SOURCES = 5


def chunk_label(chunk: dict) -> str:
    """Short badge text, like 'policy.pdf p.3' (just the name if no page)."""
    if chunk.get("page"):
        return f"{chunk['source']} p.{chunk['page']}"
    return chunk["source"]


def build_citations(text: str, chunks: list[dict]) -> tuple[str, list[dict], list[int]]:
    """Turns the [n] numbers in an answer into citations.

    Returns (answer, citations, invalid). Numbers that match a retrieved
    chunk stay in the answer and become a citation with the file name, page
    and the chunk text. A number the model made up (no such chunk was
    retrieved) is removed from the answer and listed in `invalid`.
    """
    # "[1, 2, 3]" becomes "[1][2][3]" so every number is read as a citation.
    text = _GROUP.sub(lambda m: "".join(f"[{n.strip()}]" for n in m.group(1).split(",")), text)
    by_ref = {c["ref"]: c for c in chunks}
    cited: list[int] = []
    invalid: list[int] = []

    def check(match):
        number = int(match.group(1))
        if number in by_ref:
            if number not in cited:
                cited.append(number)
            return match.group(0)
        invalid.append(number)
        return ""

    answer = _REF.sub(check, text)
    if invalid:
        answer = re.sub(r"[ \t]{2,}", " ", answer)
        answer = re.sub(r"[ \t]+([.,;:!?])", r"\1", answer).strip()

    citations = [
        {
            "ref": n,
            "source": by_ref[n]["source"],
            "page": by_ref[n]["page"],
            "label": chunk_label(by_ref[n]),
            "snippet": by_ref[n]["text"][:200],
        }
        for n in cited
    ]
    return answer, citations, invalid


def source_labels(citations: list[dict]) -> list[str]:
    labels = []
    for c in citations:
        if c["label"] not in labels:
            labels.append(c["label"])
    return labels[:MAX_SOURCES]
