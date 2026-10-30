import re

_REF = re.compile(r"\[(\d{1,2})\]")
_WORD = re.compile(r"\w+", re.UNICODE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

# A cited sentence counts as supported when at least this share of its
# meaningful words appear in the chunk it cites.
MIN_SUPPORT = 0.5

_STOP = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from",
    "has", "have", "had", "not", "but", "can", "will", "its", "their", "they",
    "them", "his", "her", "you", "your", "our", "all", "any", "than", "then",
    "also", "into", "over", "per", "who", "what", "which", "when", "where",
    "how", "such", "may", "must", "should", "would", "could", "been", "being",
    "there", "these", "those", "only", "each", "other", "about", "after",
    "before", "within", "under", "more", "most", "some", "very", "just",
}


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def _content_words(text: str) -> set[str]:
    words = set()
    for word in _WORD.findall(text.lower()):
        if word in _STOP or (len(word) < 3 and not word.isdigit()):
            continue
        words.add(_stem(word))
    return words


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def _claims(text: str) -> list[tuple[str, list[int]]]:
    """Splits an answer at its [n] markers. The text since the last marker is
    the claim that marker supports; markers with nothing between them (like
    '[1][2]') share one claim."""
    matches = list(_REF.finditer(text))
    claims = []
    start = 0
    i = 0
    while i < len(matches):
        refs = [int(matches[i].group(1))]
        end = matches[i].start()
        j = i + 1
        while j < len(matches) and not text[matches[j - 1].end():matches[j].start()].strip():
            refs.append(int(matches[j].group(1)))
            j += 1
        claims.append((text[start:end].strip().lstrip(".,;:!? ").strip(), refs))
        start = matches[j - 1].end()
        i = j
    return claims


def check_grounding(answer: str, chunks: list[dict], invalid_refs: list[int] | None = None) -> dict:
    """Checks that each cited sentence is backed by the chunk it cites.

    A sentence is supported when most of its meaningful words appear in the
    cited chunk and every number it states is in that chunk too. This is a
    word-overlap check, not a meaning check: it catches a wrong number or a
    sentence the chunk never talks about, but a paraphrase that shares few
    words with the chunk can be flagged by mistake.

    status is 'grounded', 'weak' (something is not backed up, or the model
    cited a chunk that was never retrieved) or 'unchecked' (no citations).
    """
    invalid_refs = invalid_refs or []
    by_ref = {c["ref"]: c for c in chunks}
    checks = []

    for claim, refs in _claims(answer):
        words = _content_words(claim)
        if not words:
            continue
        numbers = _numbers(claim)
        # A sentence that cites two chunks may draw from both, so its words
        # and numbers are checked against the two together.
        cited_text = " ".join(by_ref[r]["text"] for r in refs if r in by_ref)
        support = len(words & _content_words(cited_text)) / len(words)
        missing = sorted(numbers - _numbers(cited_text))
        checks.append({
            "refs": refs,
            "claim": claim[:160],
            "support": round(support, 2),
            "missing_numbers": missing,
            "ok": support >= MIN_SUPPORT and not missing,
        })

    if not checks and not invalid_refs:
        status, score = "unchecked", None
    else:
        weak = bool(invalid_refs) or any(not c["ok"] for c in checks)
        status = "weak" if weak else "grounded"
        score = round(sum(c["support"] for c in checks) / len(checks), 2) if checks else 0.0
    return {"status": status, "score": score, "checks": checks, "invalid_refs": invalid_refs}
