"""
Load test: hits a running /chat endpoint at increasing concurrency and
reports p50/p95 latency and throughput. Run it once against the app
configured with LLM_BACKEND=gemini and once with LLM_BACKEND=local
(Ollama) to compare.

Usage:
    python scripts/load_test.py --url http://localhost:8000 --email me@example.com --password secret123
    python scripts/load_test.py --concurrency 1 5 10 20 --total-requests 40
"""

import argparse
import asyncio
import time

import httpx

QUESTIONS = [
    "What is the refund policy?",
    "Summarize the uploaded contract.",
    "What's 45 * 12?",
    "Who is responsible for late deliveries?",
    "What are the termination conditions?",
]


def get_access_token(url, email, password):
    """Registers the account if it doesn't exist yet, then logs in - so
    this script works on a fresh database with zero manual setup."""
    with httpx.Client() as client:
        register_response = client.post(f"{url}/auth/register", json={"email": email, "password": password})
        if register_response.status_code not in (201, 409):  # 409 = already exists, that's fine
            register_response.raise_for_status()

        login_response = client.post(f"{url}/auth/login", json={"email": email, "password": password})
        login_response.raise_for_status()
        return login_response.json()["access_token"]


async def one_request(client, url, access_token, question):
    start = time.perf_counter()
    try:
        response = await client.post(
            f"{url}/chat",
            json={"query": question},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=60.0,
        )
        response.raise_for_status()
        ok = True
    except Exception:
        ok = False
    return time.perf_counter() - start, ok


async def run_at_concurrency(url, access_token, concurrency, total_requests):
    """Fires `total_requests` requests in batches of `concurrency` at a
    time - a simple way to control load without a separate tool like
    locust, at the cost of being less precise about sustained throughput
    than a proper load-testing tool would be."""
    latencies = []
    failures = 0
    async with httpx.AsyncClient() as client:
        for batch_start in range(0, total_requests, concurrency):
            batch_size = min(concurrency, total_requests - batch_start)
            questions = [QUESTIONS[i % len(QUESTIONS)] for i in range(batch_start, batch_start + batch_size)]
            results = await asyncio.gather(*[
                one_request(client, url, access_token, q) for q in questions
            ])
            for latency, ok in results:
                if ok:
                    latencies.append(latency)
                else:
                    failures += 1
    return latencies, failures


def percentile(data, p):
    data = sorted(data)
    if not data:
        return float("nan")
    k = int(len(data) * p / 100)
    return data[min(k, len(data) - 1)]


async def main(url, access_token, concurrencies, total_requests):
    print(f"{'concurrency':>11} | {'p50 (s)':>8} | {'p95 (s)':>8} | {'req/s':>7} | failures")
    for c in concurrencies:
        start = time.perf_counter()
        latencies, failures = await run_at_concurrency(url, access_token, c, total_requests)
        elapsed = time.perf_counter() - start
        throughput = total_requests / elapsed if elapsed > 0 else 0
        print(f"{c:>11} | {percentile(latencies, 50):>8.2f} | {percentile(latencies, 95):>8.2f} "
              f"| {throughput:>7.2f} | {failures}")

    print(
        "\nRun this again against the same app with LLM_BACKEND=local "
        "(Ollama, and the app restarted) to get the comparison row for the "
        "self-hosted backend. Gemini's approximate cost per 1000 queries "
        "is in config.py's GEMINI_*_COST_PER_MILLION settings; Ollama's "
        "cost is whatever the serving hardware costs per hour, which this "
        "script doesn't estimate - the honest cost comparison here is "
        "latency/throughput under load, not a dollar figure."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--email", default="loadtest@example.com")
    parser.add_argument("--password", default="loadtest-password-123")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--total-requests", type=int, default=20)
    args = parser.parse_args()
    token = get_access_token(args.url, args.email, args.password)
    asyncio.run(main(args.url, token, args.concurrency, args.total_requests))
