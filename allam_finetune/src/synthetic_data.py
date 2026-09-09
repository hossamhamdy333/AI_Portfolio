"""
Curation helpers for the synthetic 'analysis' examples generated in
notebooks/05_synthetic_data_engineering.ipynb.

Same idea as the guardrails layer added to Azure_RAG_Assistant (PII/format
checks before data is trusted), applied here to LLM-generated training
data instead of a live query.
"""

import re

# Matches the two-header format every real 'analysis' example uses -
# see the alpaca-format 'output' column, e.g.:
#   المجال القانوني: ...
#   النقاط الرئيسية:
#   • ...
_SCHEMA_RE = re.compile(r"المجال القانوني\s*[:：].+النقاط الرئيسية\s*[:：]", re.DOTALL)

# A crude Arabic-content check: at least 40% of non-space characters fall
# in the Arabic Unicode block. Catches empty, garbled, or wrong-language
# generations without needing a real language-ID model.
_ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")

_PII_PATTERNS = [
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),  # email
    re.compile(r"\b\d{14}\b"),  # national-ID-shaped
]


def has_valid_schema(output_text):
    """Does this output actually follow the two-header analysis format?"""
    return _SCHEMA_RE.search(output_text) is not None


def is_mostly_arabic(text, threshold=0.4):
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return False
    arabic_count = sum(1 for c in non_space if _ARABIC_RANGE.match(c))
    return (arabic_count / len(non_space)) >= threshold


def passes_length_filter(text, min_length=10, max_length=2048):
    return min_length <= len(text) <= max_length


def contains_pii(text):
    return any(pattern.search(text) for pattern in _PII_PATTERNS)


def normalize_for_dedup(text):
    """Collapse whitespace and drop diacritics-adjacent punctuation
    variance so near-identical generations hash the same way."""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[،,.:؛;]", "", text)
    return text


def word_shingles(text, n=5):
    words = text.split()
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(text_a, text_b):
    shingles_a = word_shingles(normalize_for_dedup(text_a))
    shingles_b = word_shingles(normalize_for_dedup(text_b))
    if not shingles_a or not shingles_b:
        return 0.0
    intersection = len(shingles_a & shingles_b)
    union = len(shingles_a | shingles_b)
    return intersection / union if union else 0.0


def find_near_duplicates(candidate_outputs, existing_outputs, threshold=0.7):
    """For each candidate output text, is it a near-duplicate of anything
    already in `existing_outputs` (or of an earlier candidate in the same
    batch)? Returns a set of indices into `candidate_outputs` to drop.

    O(n * m) - fine for a few hundred candidates against a few thousand
    existing rows; would need a real similarity index (e.g. MinHash/LSH)
    at web scale, which this dataset isn't at.
    """
    to_drop = set()
    kept_so_far = []
    for i, candidate in enumerate(candidate_outputs):
        is_dup = any(jaccard_similarity(candidate, other) >= threshold for other in existing_outputs)
        if not is_dup:
            is_dup = any(jaccard_similarity(candidate, other) >= threshold for other in kept_so_far)
        if is_dup:
            to_drop.add(i)
        else:
            kept_so_far.append(candidate)
    return to_drop


def curate_batch(records, existing_outputs, min_length=10, max_length=2048, dedup_threshold=0.7):
    """Run every check on a batch of freshly-generated records. Each
    record is a dict with at least an "output" key - any other keys
    (e.g. "input", "system", "instruction") are carried through untouched,
    so nothing gets separated from the rest of its own row partway
    through the funnel, however many fields a row has.

    Returns (kept_records, funnel): kept_records is the list of surviving
    dicts, unchanged. funnel counts how many rows were dropped at each
    step, in order - a standard dataset-engineering artifact, not just a
    final pass/fail count."""
    funnel = {"input": len(records), "schema_fail": 0, "length_fail": 0,
              "language_fail": 0, "pii_fail": 0, "near_duplicate": 0}

    stage_1 = []
    for record in records:
        output_text = record["output"]
        if not has_valid_schema(output_text):
            funnel["schema_fail"] += 1
        elif not passes_length_filter(output_text, min_length, max_length):
            funnel["length_fail"] += 1
        elif not is_mostly_arabic(output_text):
            funnel["language_fail"] += 1
        elif contains_pii(output_text):
            funnel["pii_fail"] += 1
        else:
            stage_1.append(record)

    stage_1_outputs = [r["output"] for r in stage_1]
    drop_indices = find_near_duplicates(stage_1_outputs, existing_outputs, dedup_threshold)
    funnel["near_duplicate"] = len(drop_indices)
    kept_records = [r for i, r in enumerate(stage_1) if i not in drop_indices]
    funnel["kept"] = len(kept_records)
    return kept_records, funnel
