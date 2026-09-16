"""
A simple per-IP rate limit: at most RATE_LIMIT_MAX_REQUESTS questions per
RATE_LIMIT_WINDOW_SECONDS, fixed window (not sliding - simpler, and the
difference doesn't matter at this traffic scale).

In-memory, not backed by the database or Redis. Real, honest limitation
worth stating plainly: this resets if the process restarts, and doesn't
share state across multiple instances if this ever gets horizontally
scaled. Fine for a single-instance demo behind one Azure Container App
replica; would need Redis (or the database, though that's a write per
request for something this cheap to compute) if that ever changes.
"""

import time

import config

# ip_address -> list of unix timestamps of requests within the current window
_request_log: dict[str, list[float]] = {}


def is_rate_limited(ip_address: str) -> bool:
    now = time.time()
    window_start = now - config.RATE_LIMIT_WINDOW_SECONDS

    timestamps = _request_log.get(ip_address, [])
    timestamps = [t for t in timestamps if t > window_start]  # drop anything outside the window

    limited = len(timestamps) >= config.RATE_LIMIT_MAX_REQUESTS

    if not limited:
        timestamps.append(now)
    _request_log[ip_address] = timestamps

    return limited


def reset_all():
    """Test-only - clears every IP's request history between test runs."""
    _request_log.clear()
