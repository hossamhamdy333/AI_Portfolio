"""
Guardrails: input/output safety checks that run before a query reaches the
agent, and before an answer reaches the user.

Three checks, each heuristic (regex-based), not a learned classifier -
good enough to demonstrate the layer and to catch the adversarial test set
in tests/adversarial_prompts.json, not a production-grade moderation
system:

- PII redaction: strips emails, phone numbers, and national-ID-shaped
  numbers out of a query before it's sent to the LLM or written to a log.
- Prompt injection detection: flags text trying to override the system
  prompt or exfiltrate it (e.g. "ignore previous instructions").
- Output moderation: flags a small set of disallowed categories in the
  agent's own answer before it's returned to the user.
"""

import re

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")
_NATIONAL_ID_RE = re.compile(r"\b\d{14}\b")  # e.g. Egyptian national ID shape

_INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|prior|above) instructions",
    r"disregard (all|any|the)? ?(previous|prior|above) (instructions|rules)",
    r"reveal (your|the) system prompt",
    r"you are now",
    r"new instructions?:",
    r"act as (if|though)",
    r"pretend (you are|to be)",
    r"jailbreak",
    r"developer mode",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_DISALLOWED_OUTPUT_PATTERNS = [
    r"here('s| is) how to (make|build|synthesize) (a bomb|explosives|malware)",
]
_DISALLOWED_OUTPUT_RE = re.compile("|".join(_DISALLOWED_OUTPUT_PATTERNS), re.IGNORECASE)


def redact_pii(text):
    """Replace anything PII-shaped with a placeholder.
    Returns (redacted_text, number_of_redactions)."""
    text, n_email = _EMAIL_RE.subn("[EMAIL]", text)
    text, n_phone = _PHONE_RE.subn("[PHONE]", text)
    text, n_id = _NATIONAL_ID_RE.subn("[ID_NUMBER]", text)
    return text, n_email + n_phone + n_id


def check_prompt_injection(text):
    """(is_suspicious, matched_pattern)."""
    match = _INJECTION_RE.search(text)
    return (match is not None, match.group(0) if match else None)


def check_output(text):
    """(is_disallowed, matched_pattern) for the agent's own answer."""
    match = _DISALLOWED_OUTPUT_RE.search(text)
    return (match is not None, match.group(0) if match else None)


def guard_input(text):
    """Run every input-side check. Returns what the caller needs to log
    and act on: the redacted text to actually send to the LLM, whether it
    looks like a prompt injection attempt, and whether to block outright."""
    redacted_text, pii_count = redact_pii(text)
    is_injection, injection_match = check_prompt_injection(text)
    return {
        "redacted_text": redacted_text,
        "pii_redactions": pii_count,
        "is_injection": is_injection,
        "injection_match": injection_match,
        "blocked": is_injection,
    }


def guard_output(text):
    is_disallowed, match = check_output(text)
    return {
        "text": "I can't help with that." if is_disallowed else text,
        "blocked": is_disallowed,
        "match": match,
    }
