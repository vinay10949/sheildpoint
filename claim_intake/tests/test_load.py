"""SP-203 AC: Load test — 100 concurrent claims via API, measure P99 latency.

This test uses the in-process API server pattern from run_load_test.py.
Marked `slow` + `load` so it can be deselected during quick test runs::

    pytest -m "not slow"        # skip load tests
    pytest -m load              # run only load tests
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import httpx
import pytest
import uvicorn


@pytest.fixture(scope="module")
def api_server():
    """Start the intake API on a free port for the duration of the module."""
    from claim_intake.api import app
    from claim_intake.store import reset_stores

    port = 8011
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for the server to come up.
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.1)
    else:
        raise RuntimeError("API server did not come up")

    # Reset stores before the test.
    reset_stores()

    yield base_url

    server.should_exit = True
    thread.join(timeout=5.0)


def _make_payload(idx: int) -> dict[str, Any]:
    # Generate names from a pool so the validator always accepts them.
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
            f"Load test claim number {idx}. Wind damage to roof shingles."
        ),
        "amount_claimed": 1250.00 + idx,
        "request_id": f"REQ-LOAD-{idx:04d}",
    }


@pytest.mark.slow
@pytest.mark.load
class TestLoad100Concurrent:
    """SP-203 AC: 100 concurrent claims via API, P99 latency measured."""

    def test_100_concurrent_claims_p99_under_30s(self, api_server):
        """Submit 100 claims at concurrency=100, assert P99 < 30s and all
        claims are accepted."""
        count = 100
        concurrency = 100
        base_url = api_server

        async def runner() -> Any:
            semaphore = asyncio.Semaphore(concurrency)
            latencies: list[float] = []
            accepted = 0
            errors: list[str] = []

            async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
                async def one(idx: int) -> None:
                    nonlocal accepted
                    async with semaphore:
                        start = time.monotonic()
                        try:
                            r = await client.post(
                                "/intake/claims", json=_make_payload(idx),
                            )
                            elapsed = time.monotonic() - start
                            latencies.append(elapsed)
                            if r.status_code == 200:
                                body = r.json()
                                if body.get("status") == "accepted":
                                    accepted += 1
                                else:
                                    errors.append(
                                        f"idx={idx} status={body.get('status')}"
                                    )
                            else:
                                errors.append(
                                    f"idx={idx} http={r.status_code} body={r.text[:200]}"
                                )
                        except Exception as exc:
                            errors.append(f"idx={idx} exception={exc}")

                await asyncio.gather(*[one(i) for i in range(count)])

            latencies.sort()
            n = len(latencies)
            return {
                "count": count,
                "accepted": accepted,
                "errors": errors,
                "p50": latencies[int(n * 0.50)],
                "p95": latencies[int(n * 0.95)],
                "p99": latencies[int(n * 0.99)],
                "max": latencies[-1],
            }

        result = asyncio.run(runner())

        # Assertions.
        assert result["accepted"] == count, (
            f"Only {result['accepted']}/{count} claims accepted. "
            f"Errors: {result['errors'][:5]}"
        )
        assert result["p99"] < 30.0, (
            f"P99 latency = {result['p99']:.3f}s exceeds 30s AC. "
            f"P50={result['p50']:.3f}s P95={result['p95']:.3f}s "
            f"max={result['max']:.3f}s"
        )
        # Sanity: max should not be more than 5x P50 (no degenerate tail).
        # 5x is generous — typical well-behaved servers are <2x.
        assert result["max"] < max(5.0, result["p50"] * 5), (
            f"max ({result['max']:.3f}s) is more than 5x P50 "
            f"({result['p50']:.3f}s) — indicates a degenerate tail."
        )

    def test_100_concurrent_at_concurrency_50(self, api_server):
        """Also verify at concurrency=50 (more realistic for a single worker).

        The AC says "100 concurrent claims" — we interpret this as 100
        claims submitted concurrently, with the server's natural
        concurrency limit applying. Running at concurrency=50 verifies the
        server handles partial parallelism without errors.
        """
        from claim_intake.store import reset_stores
        reset_stores()

        count = 100
        concurrency = 50
        base_url = api_server

        async def runner() -> Any:
            semaphore = asyncio.Semaphore(concurrency)
            accepted = 0
            async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
                async def one(idx: int) -> None:
                    nonlocal accepted
                    async with semaphore:
                        r = await client.post(
                            "/intake/claims", json=_make_payload(idx),
                        )
                        if r.status_code == 200 and r.json().get("status") == "accepted":
                            accepted += 1
                await asyncio.gather(*[one(i) for i in range(count)])
            return accepted

        accepted = asyncio.run(runner())
        assert accepted == count
