#!/usr/bin/env python3
"""
inference-bench.py — measure LM Studio throughput (tokens/sec) and latency.

Usage:
    python3 scripts/inference-bench.py \\
        --base-url http://localhost:1234/v1 \\
        --model qwen3.6-35b-a3b-q4_k_m \\
        --prompt "Hello, what is homeowner's insurance?" \\
        --max-tokens 200 \\
        --runs 3

    # Smoke-test integration: print only the average tokens/sec
    python3 scripts/inference-bench.py ... --print-tps-only

    # Quiet (no per-run output)
    python3 scripts/inference-bench.py ... --quiet

Reports:
    - Per-run: latency (ms), tokens generated, throughput (tok/s)
    - Aggregate: mean / median / min / max throughput

Exits 0 on success, 1 on any HTTP/parse error.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from urllib import error, request


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LM Studio throughput benchmark")
    p.add_argument("--base-url", required=True, help="LM Studio base URL, e.g. http://localhost:1234/v1")
    p.add_argument("--model", required=True, help="Model id (must match GET /v1/models)")
    p.add_argument("--prompt", default="Hello, what is homeowner's insurance?", help="Prompt text")
    p.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate per run")
    p.add_argument("--runs", type=int, default=3, help="Number of runs to average")
    p.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    p.add_argument("--api-key", default=os.environ.get("LM_STUDIO_API_KEY", "lm-studio"))
    p.add_argument("--quiet", action="store_true", help="Suppress per-run output")
    p.add_argument("--print-tps-only", action="store_true", help="Print only mean tok/s (for shell scripts)")
    return p.parse_args()


def bench_once(base_url: str, model: str, prompt: str, max_tokens: int,
               temperature: float, api_key: str) -> tuple[float, int, str]:
    """Run one inference call. Return (latency_ms, tokens_generated, content)."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode("utf-8")

    req = request.Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    start = time.perf_counter()
    try:
        with request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except error.URLError as e:
        print(f"[bench] HTTP error: {e}", file=sys.stderr)
        sys.exit(1)
    latency_ms = (time.perf_counter() - start) * 1000

    try:
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
        # Use LM Studio's usage if present; otherwise estimate by whitespace tokenization.
        tokens = data.get("usage", {}).get("completion_tokens") or len(content.split())
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"[bench] parse error: {e}; body[:200]={body[:200]}", file=sys.stderr)
        sys.exit(1)

    return latency_ms, tokens, content


def main() -> int:
    args = parse_args()

    # Warmup (one call, not counted) — first call after load JIT-compiles kernels.
    if not args.print_tps_only:
        print(f"[bench] warmup run...", file=sys.stderr)
    try:
        bench_once(args.base_url, args.model, args.prompt,
                   max_tokens=8, temperature=args.temperature, api_key=args.api_key)
    except SystemExit:
        # Warmup failure is fatal — surface it.
        raise

    tps_samples: list[float] = []
    for i in range(args.runs):
        lat_ms, toks, content = bench_once(
            args.base_url, args.model, args.prompt,
            max_tokens=args.max_tokens, temperature=args.temperature, api_key=args.api_key,
        )
        tps = toks / (lat_ms / 1000.0) if lat_ms > 0 else 0.0
        tps_samples.append(tps)
        if not args.quiet and not args.print_tps_only:
            preview = content[:60].replace("\n", " ")
            print(f"  run {i+1}/{args.runs}: {lat_ms:.0f}ms, {toks} toks, {tps:.1f} tok/s — '{preview}...'")

    mean_tps = statistics.mean(tps_samples)
    if args.print_tps_only:
        # Print only the mean tok/s, formatted to 1 decimal.
        print(f"{mean_tps:.1f}")
        return 0

    print()
    print(f"  Runs:        {args.runs}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Mean tps:    {mean_tps:.1f}")
    if args.runs > 1:
        print(f"  Median tps:  {statistics.median(tps_samples):.1f}")
        print(f"  Min tps:     {min(tps_samples):.1f}")
        print(f"  Max tps:     {max(tps_samples):.1f}")
        print(f"  Stdev tps:   {statistics.stdev(tps_samples):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
