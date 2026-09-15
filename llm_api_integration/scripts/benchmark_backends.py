"""
Load test: hits a running /analyze endpoint at increasing concurrency and
reports p50/p95 latency and throughput. Run it three times - once per
config.yaml backend.provider setting (gemini, ollama, vllm), restarting
the app between runs - and compare the three printed tables directly.

/analyze is the benchmark target because it's a single request/response
round trip (no streaming to consume), the same shape regardless of which
backend answered it.

Usage:
    python scripts/benchmark_backends.py --url http://localhost:8000
    python scripts/benchmark_backends.py --concurrency 1 5 10 20 --total-requests 40
"""

import argparse
import asyncio
import time

import httpx

TEXTS = [
    "I absolutely love this product, best purchase all year!",
    "This was a complete waste of money, extremely disappointed.",
    "It's fine, does what it says, nothing special.",
    "Terrible customer service, I want a refund immediately.",
    "Pretty good overall, would recommend to a friend.",
]


async def one_request(client, url, text):
    start = time.perf_counter()
    try:
        response = await client.post(f"{url}/analyze", json={"text": text}, timeout=60.0)
        response.raise_for_status()
        ok = True
    except Exception:
        ok = False
    return time.perf_counter() - start, ok


async def run_at_concurrency(url, concurrency, total_requests):
    latencies = []
    failures = 0
    async with httpx.AsyncClient() as client:
        for batch_start in range(0, total_requests, concurrency):
            batch_size = min(concurrency, total_requests - batch_start)
            texts = [TEXTS[i % len(TEXTS)] for i in range(batch_start, batch_start + batch_size)]
            results = await asyncio.gather(*[one_request(client, url, t) for t in texts])
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


async def main(url, concurrencies, total_requests):
    health = None
    async with httpx.AsyncClient() as client:
        try:
            health = (await client.get(f"{url}/health", timeout=10.0)).json()
        except Exception:
            pass

    if health:
        print(f"Backend: {health.get('backend')}  Model: {health.get('model')}\n")

    print(f"{'concurrency':>11} | {'p50 (s)':>8} | {'p95 (s)':>8} | {'req/s':>7} | failures")
    for c in concurrencies:
        start = time.perf_counter()
        latencies, failures = await run_at_concurrency(url, c, total_requests)
        elapsed = time.perf_counter() - start
        throughput = total_requests / elapsed if elapsed > 0 else 0
        print(f"{c:>11} | {percentile(latencies, 50):>8.2f} | {percentile(latencies, 95):>8.2f} "
              f"| {throughput:>7.2f} | {failures}")

    print(
        "\nRun this again after changing config.yaml's backend.provider (and restarting "
        "the app) to get the comparison row for a different backend. Gemini's approximate "
        "cost per request is tracked in MLflow (see tracking.py) - the self-hosted backends' "
        "cost is whatever the serving hardware costs per hour, which this script doesn't "
        "estimate; the honest comparison here is latency/throughput, not a dollar figure."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--total-requests", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.concurrency, args.total_requests))
