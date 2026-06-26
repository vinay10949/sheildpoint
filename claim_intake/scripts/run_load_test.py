#!/usr/bin/env python3
"""
SP-203 Load Test: 100 concurrent claims via API, measure P99 latency.

Usage::

    # Start the intake API in a separate terminal first:
    INTAKE_API_PORT=8001 python -m claim_intake.api

    # Then run the load test:
    python scripts/run_load_test.py --count 100 --concurrency 50

Or, to spin up the API in-process for a self-contained test::

    python scripts/run_load_test.py --count 100 --in-process

Outputs a JSON summary + per-claim latencies to stdout. Exit code is 0 if
all claims succeeded and P99 < 30s, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Sample payload generator — vary the policyholder name so each claim gets
# a unique claim_id (the store is in-memory, so collisions don't matter,
# but it makes the test more realistic).
# ---------------------------------------------------------------------------
def make_payload(idx: int) -> dict[str, Any]:
    # Generate names from a pool so the validator always accepts them
    # (no pure-numeric tokens, which would fail the name regex).
    first_names = ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace",
                   "Heidi", "Ivan", "Judy"]
    last_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta",
                  "Eta", "Theta", "Iota", "Kappa"]
    name = f"{first_names[idx % len(first_names)]} {last_names[idx % len(last_names)]}"
    return {
        "policyholder_name": name,
        "policy_id": f"HO-2024-{idx:05d}",
        "claim_type": "homeowners",
        "date_of_loss": "2026-03-14",
        "damage_description": (
            f"Load test claim number {idx}. Wind damage to roof shingles "
            f"during severe thunderstorm. Approximately 30% blown off."
        ),
        "amount_claimed": 1250.00 + idx,
        "request_id": f"REQ-LOAD-{idx:04d}",
    }


# ---------------------------------------------------------------------------
# Single claim submission
# ---------------------------------------------------------------------------
async def submit_one(
    client: httpx.AsyncClient, payload: dict[str, Any],
) -> tuple[int, float, dict[str, Any]]:
    start = time.monotonic()
    r = await client.post("/intake/claims", json=payload)
    elapsed = time.monotonic() - start
    body = r.json() if r.status_code == 200 else {"error": r.text, "status_code": r.status_code}
    return r.status_code, elapsed, body


# ---------------------------------------------------------------------------
# Concurrency-bounded runner
# ---------------------------------------------------------------------------
async def run_load_test(
    base_url: str, *, count: int, concurrency: int,
) -> dict[str, Any]:
    """Run ``count`` concurrent claims at ``concurrency`` parallelism."""
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    statuses: list[int] = []
    state = {"accepted": 0, "in_review": 0, "rejected": 0}
    errors: list[dict[str, Any]] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        async def bounded_submit(idx: int) -> None:
            async with semaphore:
                status, latency, body = await submit_one(client, make_payload(idx))
                latencies.append(latency)
                statuses.append(status)
                if status != 200:
                    errors.append({"idx": idx, "status": status, "body": body})
                    return
                # Tally by intake status
                intake_status = body.get("status")
                if intake_status == "accepted":
                    state["accepted"] += 1
                elif intake_status == "in_review":
                    state["in_review"] += 1
                elif intake_status == "rejected":
                    state["rejected"] += 1

        # Fire all submissions concurrently (bounded by the semaphore).
        overall_start = time.monotonic()
        await asyncio.gather(*[bounded_submit(i) for i in range(count)])
        overall_elapsed = time.monotonic() - overall_start

    # Compute percentiles.
    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)] if n else 0.0
    p95 = latencies[int(n * 0.95)] if n else 0.0
    p99 = latencies[int(n * 0.99)] if n else 0.0
    mean = statistics.mean(latencies) if latencies else 0.0

    return {
        "count": count,
        "concurrency": concurrency,
        "overall_elapsed_sec": overall_elapsed,
        "throughput_claims_per_sec": count / overall_elapsed if overall_elapsed else 0,
        "latency_sec": {
            "min": min(latencies) if latencies else 0,
            "mean": mean,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "max": max(latencies) if latencies else 0,
        },
        "outcomes": {
            "accepted": state["accepted"],
            "in_review": state["in_review"],
            "rejected": state["rejected"],
            "http_errors": sum(1 for s in statuses if s != 200),
        },
        "errors": errors[:10],  # first 10 errors for inspection
        "p99_under_30s": p99 < 30.0,
        "all_succeeded": len(errors) == 0 and state["accepted"] == count,
    }


# ---------------------------------------------------------------------------
# In-process API server (for self-contained load test)
# ---------------------------------------------------------------------------
def start_in_process_api(port: int) -> Any:
    """Start the intake API in a background thread."""
    import threading
    import uvicorn
    from claim_intake.api import app
    from claim_intake.config import IntakeConfig

    cfg = IntakeConfig.from_env()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for the server to come up.
    import time as _time
    deadline = _time.monotonic() + 10.0
    import httpx as _httpx
    while _time.monotonic() < deadline:
        try:
            r = _httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if r.status_code == 200:
                return server, thread
        except Exception:
            pass
        _time.sleep(0.1)
    raise RuntimeError(f"API server did not come up on port {port}")


def stop_in_process_api(server: Any) -> None:
    server.should_exit = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="SP-203 Claim Intake load test")
    p.add_argument("--base-url", default="http://127.0.0.1:8001",
                   help="Intake API base URL")
    p.add_argument("--count", type=int, default=100,
                   help="Number of claims to submit")
    p.add_argument("--concurrency", type=int, default=50,
                   help="Maximum concurrent submissions")
    p.add_argument("--in-process", action="store_true",
                   help="Start the API in-process (no separate server needed)")
    p.add_argument("--port", type=int, default=8009,
                   help="Port for in-process API (default: 8009)")
    p.add_argument("--json", action="store_true",
                   help="Output raw JSON instead of pretty-printed summary")
    args = p.parse_args()

    server = None
    base_url = args.base_url
    if args.in_process:
        print(f"[load-test] Starting in-process API on port {args.port}...")
        server, _ = start_in_process_api(args.port)
        base_url = f"http://127.0.0.1:{args.port}"
        # Reset stores before the test.
        try:
            httpx.post(f"{base_url}/review/queue")  # noop, just to confirm reachability
        except Exception:
            pass

    try:
        print(f"[load-test] Submitting {args.count} claims at concurrency={args.concurrency}...")
        result = asyncio.run(run_load_test(
            base_url, count=args.count, concurrency=args.concurrency,
        ))
    finally:
        if server is not None:
            stop_in_process_api(server)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print()
        print("=" * 70)
        print("SP-203 Load Test Results")
        print("=" * 70)
        print(f"  Claims submitted:   {result['count']}")
        print(f"  Concurrency:        {result['concurrency']}")
        print(f"  Total elapsed:      {result['overall_elapsed_sec']:.2f}s")
        print(f"  Throughput:         {result['throughput_claims_per_sec']:.1f} claims/sec")
        print()
        print("Latency (seconds):")
        print(f"  min:   {result['latency_sec']['min']:.3f}")
        print(f"  mean:  {result['latency_sec']['mean']:.3f}")
        print(f"  P50:   {result['latency_sec']['p50']:.3f}")
        print(f"  P95:   {result['latency_sec']['p95']:.3f}")
        print(f"  P99:   {result['latency_sec']['p99']:.3f}  (AC: < 30.000)")
        print(f"  max:   {result['latency_sec']['max']:.3f}")
        print()
        print("Outcomes:")
        for k, v in result["outcomes"].items():
            print(f"  {k}: {v}")
        if result["errors"]:
            print()
            print(f"First {len(result['errors'])} errors:")
            for err in result["errors"]:
                print(f"  idx={err['idx']} status={err['status']} body={err['body']}")
        print()
        print("=" * 70)
        if result["p99_under_30s"] and result["all_succeeded"]:
            print("PASS: P99 < 30s and all claims succeeded.")
            return 0
        else:
            if not result["p99_under_30s"]:
                print(f"FAIL: P99 = {result['latency_sec']['p99']:.3f}s exceeds 30s AC.")
            if not result["all_succeeded"]:
                print("FAIL: not all claims succeeded (see errors above).")
            return 1


if __name__ == "__main__":
    sys.exit(main())
