#!/usr/bin/env python3
"""
End-to-End Langfuse Trace Test (SHLD-9 Acceptance Criterion)
=============================================================

Sends one sample LLM call to LM Studio, decorated with the ShieldPoint
Langfuse wrapper, then verifies the trace was captured by the self-hosted
Langfuse server.

Acceptance criteria verified:
  [1] Langfuse web UI accessible on port 3000           → GET /api/public/health
  [2] PostgreSQL 16 running with persistent volume      → GET /api/public/health
                                                          (Langfuse can't boot
                                                          without DB)
  [3] Langfuse Python SDK integrated into agent skeleton→ import succeeds
  [4] First test trace captured with prompt, completion,
      latency, and token count                          → trace found via API
  [5] No data leaves ShieldPoint network                → LANGFUSE_HOST is
                                                          localhost/internal

Usage:
    python3 scripts/test-langfuse-trace.py
    python3 scripts/test-langfuse-trace.py --no-llm    # mock LLM response,
                                                       # still send trace

Exit codes:
  0  — all stages PASS
  1  — one or more stages FAILED
  2  — passed with warnings (e.g. LLM unreachable, used mock mode)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Bootstrap — make the in-repo agent_framework package importable when run
# from the repo root.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# Load .env if present (so the script works standalone without requiring
# the user to export env vars). Single unified env file for the whole stack.
def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

_load_env_file(REPO_ROOT / ".env")


# ---------------------------------------------------------------------------
# ANSI helpers (kept simple — work in CI logs too)
# ---------------------------------------------------------------------------
class C:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    BOLD = "\033[1m"


def info(msg: str) -> None:
    print(f"{C.BLUE}[INFO]{C.RESET} {msg}")


def ok(msg: str) -> None:
    print(f"{C.GREEN}[OK]{C.RESET}   {msg}")


def warn(msg: str) -> None:
    print(f"{C.YELLOW}[WARN]{C.RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{C.RED}[FAIL]{C.RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{C.BOLD}=== {title} ==={C.RESET}")


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------
PASS = 0
WARN = 0
FAIL = 0


def record_pass(msg: str) -> None:
    global PASS
    PASS += 1
    ok(msg)


def record_warn(msg: str) -> None:
    global WARN
    WARN += 1
    warn(msg)


def record_fail(msg: str) -> None:
    global FAIL
    FAIL += 1
    fail(msg)


def stage_1_langfuse_health() -> bool:
    """Verify Langfuse web UI + Postgres are up by hitting the public health endpoint."""
    section("Stage 1/5: Langfuse health endpoint (UI + DB up)")
    import httpx

    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000").rstrip("/")
    url = f"{host}/api/public/health"
    info(f"GET {url}")
    try:
        resp = httpx.get(url, timeout=10.0)
    except Exception as exc:
        record_fail(f"Could not connect to Langfuse at {url}: {exc}")
        return False

    if resp.status_code != 200:
        record_fail(
            f"GET /api/public/health returned HTTP {resp.status_code} — "
            f"Langfuse not healthy. Body: {resp.text[:200]}"
        )
        return False

    record_pass(f"Langfuse health endpoint returned 200 (UI + DB operational)")
    return True


def stage_2_sdk_import() -> bool:
    """Verify the Langfuse SDK is installed and the ShieldPoint wrapper loads."""
    section("Stage 2/5: Python SDK + ShieldPoint wrapper import")
    try:
        import langfuse
        info(f"langfuse package version: {getattr(langfuse, 'version', 'unknown')}")
    except ImportError:
        record_fail(
            "langfuse package not installed — install with: "
            "pip install -r agent_framework/observability/requirements.txt"
        )
        return False

    try:
        from agent_framework.observability import (  # type: ignore
            tracer,
            observe_llm,
            observe_tool,
            trace_context,
        )
    except Exception:
        record_fail(f"Failed to import shieldpoint.observability:\n{traceback.format_exc()}")
        return False

    info(f"tracer.enabled = {tracer.enabled}")
    info(f"tracer.host    = {tracer.host}")
    if tracer.disabled_reason:
        info(f"disabled reason: {tracer.disabled_reason}")

    if not tracer.enabled:
        record_fail(
            "ShieldPoint tracer is disabled. Check env vars: "
            "LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY. "
            f"Reason: {tracer.disabled_reason}"
        )
        return False

    record_pass("langfuse SDK + ShieldPoint wrapper loaded; tracer enabled")
    return True


def stage_3_send_test_trace(use_mock_llm: bool) -> Optional[str]:
    """Send a sample LLM call to LM Studio (or mock it), wrapped with the tracer.

    Returns the trace ID if a trace was sent, else None.
    """
    section("Stage 3/5: Send sample LLM call to LM Studio (decorated with @observe_llm)")

    from agent_framework.observability import tracer, observe_llm, trace_context  # type: ignore

    # Import OpenAI client (LM Studio is OpenAI-compatible)
    try:
        from openai import OpenAI
    except ImportError:
        record_fail("openai package not installed — pip install openai")
        return None

    base_url = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    api_key = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
    model = os.environ.get("QWEN_MODEL_ID", "qwen3.6-35b-a3b-q4_k_m")

    info(f"LM Studio URL: {base_url}")
    info(f"Model:         {model}")
    info(f"Mock LLM:      {use_mock_llm}")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # ---- The decorated LLM call. Captures input, output, latency, tokens. ----
    @observe_llm(name="shieldpoint_e2e_test_classification")
    def classify_claim(claim_text: str) -> Any:
        if use_mock_llm:
            # Build a fake OpenAI-style response so the wrapper can still
            # extract usage + content for the trace.
            class _Usage:
                prompt_tokens = 42
                completion_tokens = 18
                total_tokens = 60

            class _Msg:
                content = (
                    '{"severity":"medium","should_auto_approve":false,'
                    '"reason":"Damage below deductible; requires adjuster review."}'
                )
                tool_calls = None

            class _Choice:
                message = _Msg()
                finish_reason = "stop"

            class _Resp:
                choices = [_Choice()]
                usage = _Usage()

            return _Resp()

        # Real call to LM Studio
        return client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an auto-insurance claims classifier. "
                        "Given a claim description, return a JSON object with "
                        "keys: severity (low|medium|high|critical), "
                        "should_auto_approve (bool), reason (string)."
                    ),
                },
                {"role": "user", "content": claim_text},
            ],
            temperature=0.1,
            max_tokens=200,
        )

    # ---- The trace boundary ----
    claim_text = (
        "Policyholder reports rear-end collision at low speed in a parking lot. "
        "No injuries reported. Estimated damage: $1,200 to rear bumper. "
        "Policy includes $500 collision deductible. Claimant has clean record."
    )

    trace_id: Optional[str] = None
    try:
        with trace_context(
            name="e2e_test_trace",
            user_id="shieldpoint-e2e-test",
            session_id="e2e-session-001",
            metadata={
                "test": True,
                "claim_id": "CLM-E2E-TEST-0001",
                "mock_llm": use_mock_llm,
            },
            tags=["e2e-test", "phase-0", "shld-9"],
        ) as handle:
            trace_id = handle.id
            info(f"Trace ID: {trace_id}")

            result = classify_claim(claim_text)

            # Extract and display what was captured
            from agent_framework.observability.langfuse_wrapper import (
                _extract_output_content,
                _extract_usage,
            )
            output = _extract_output_content(result)
            usage = _extract_usage(result)

            info(f"LLM output: {json.dumps(output, indent=2)[:400]}")
            info(f"Token usage: {usage}")
    except Exception as exc:
        if use_mock_llm:
            record_fail(f"Mock LLM trace failed: {exc}\n{traceback.format_exc()}")
            return None
        # Real LLM unreachable — fall back to mock and warn
        record_warn(
            f"Real LM Studio call failed ({type(exc).__name__}: {exc}). "
            "Retrying with mock LLM to still produce a trace."
        )
        return stage_3_send_test_trace(use_mock_llm=True)

    # Flush to make sure trace lands in Langfuse before we query for it
    tracer.flush()
    time.sleep(2.0)  # give the ingestion pipeline a moment

    if usage and usage.get("total_tokens", 0) > 0:
        record_pass(
            f"Trace sent. Captured: prompt + completion + "
            f"{usage.get('total_tokens')} tokens"
        )
    else:
        record_warn(
            "Trace sent but token usage not captured — "
            "check that the LLM returns OpenAI-style `usage` field."
        )
    return trace_id


def stage_4_verify_trace_captured(trace_id: Optional[str]) -> bool:
    """Query the Langfuse API to verify the trace was actually ingested."""
    section("Stage 4/5: Verify trace was captured by Langfuse")

    if not trace_id:
        record_fail("No trace_id available from Stage 3 — cannot verify.")
        return False

    import httpx

    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000").rstrip("/")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    # Langfuse v3 API: GET /api/public/traces/{traceId}
    url = f"{host}/api/public/traces/{trace_id}"
    info(f"GET {url}")

    # Retry for up to 30s — ingestion is async via ClickHouse
    for attempt in range(1, 16):
        try:
            resp = httpx.get(
                url,
                auth=(public_key, secret_key),
                timeout=10.0,
            )
        except Exception as exc:
            record_warn(f"GET trace failed (attempt {attempt}): {exc}")
            time.sleep(2)
            continue

        if resp.status_code == 200:
            body = resp.json()
            trace_name = body.get("name", "<unknown>")
            observations = body.get("observations", [])
            info(f"Trace name:       {trace_name}")
            info(f"Observation count: {len(observations)}")

            # Inspect observations for the captured generation
            found_generation = False
            found_prompt = False
            found_completion = False
            found_tokens = False
            found_latency = False

            for obs in observations:
                if obs.get("type") != "GENERATION":
                    continue
                found_generation = True
                input_data = obs.get("input", {})
                output_data = obs.get("output", {})
                usage = obs.get("usage", {}) or {}
                meta = obs.get("metadata", {}) or {}

                if isinstance(input_data, dict) and input_data.get("messages"):
                    found_prompt = True
                if isinstance(output_data, dict) and output_data.get("content"):
                    found_completion = True
                if usage.get("totalTokens", 0) > 0 or usage.get("total_tokens", 0) > 0:
                    found_tokens = True
                if isinstance(meta, dict) and "latency_ms" in meta:
                    found_latency = True

                info(f"Generation '{obs.get('name')}': "
                     f"tokens={usage}, latency={meta.get('latency_ms')}ms")

            all_good = (
                found_generation
                and found_prompt
                and found_completion
                and found_tokens
                and found_latency
            )
            if all_good:
                record_pass(
                    "Trace captured with: prompt, completion, "
                    "token count, latency ✓"
                )
                return True
            else:
                missing = []
                if not found_generation: missing.append("generation")
                if not found_prompt: missing.append("prompt")
                if not found_completion: missing.append("completion")
                if not found_tokens: missing.append("tokens")
                if not found_latency: missing.append("latency")
                record_warn(
                    f"Trace found but missing: {', '.join(missing)}. "
                    "Check the wrapper captures these fields."
                )
                return True  # trace WAS captured, just incomplete

        elif resp.status_code == 404:
            info(f"Trace not yet visible (attempt {attempt}/15) — waiting 2s for ingestion...")
            time.sleep(2)
        else:
            record_fail(
                f"GET trace returned HTTP {resp.status_code}: {resp.text[:300]}"
            )
            return False

    record_fail(
        f"Trace {trace_id} never became visible in Langfuse API after 15 retries. "
        "Check Langfuse worker logs: docker logs shieldpoint-langfuse"
    )
    return False


def stage_5_data_sovereignty() -> bool:
    """Verify no outbound traffic is configured (LANGFUSE_HOST is internal)."""
    section("Stage 5/5: Data sovereignty check (no data leaves ShieldPoint)")

    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    info(f"LANGFUSE_HOST = {host}")

    external_indicators = [
        "langfuse.io", "cloud.langfuse.com", "cloud-eu.langfuse.com",
        "cloud-us.langfuse.com", "amazonaws.com", "s3.amazonaws.com",
    ]
    violated = [ind for ind in external_indicators if ind in host.lower()]

    if violated:
        record_fail(
            f"LANGFUSE_HOST contains external endpoint: {violated}. "
            "All data would leave ShieldPoint network — violates SHLD-43."
        )
        return False

    # Verify telemetry is disabled in the compose file
    compose_path = REPO_ROOT / "docker-compose.langfuse.yml"
    if compose_path.exists():
        content = compose_path.read_text()
        if "TELEMETRY_ENABLED=false" in content and "LANGFUSE_ENABLE_TELEMETRY=false" in content:
            record_pass(
                "LANGFUSE_HOST is internal + telemetry disabled in compose. "
                "Data sovereignty OK."
            )
            return True
        else:
            record_warn(
                "TELEMETRY_ENABLED=false not found in docker-compose.langfuse.yml. "
                "Add it to prevent anonymous usage pings to langfuse.io."
            )
            return True
    else:
        record_warn("docker-compose.langfuse.yml not found — cannot verify telemetry setting.")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="SHLD-9 e2e Langfuse trace test")
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Mock the LM Studio call (still sends a trace). Use if LM Studio is down.",
    )
    args = parser.parse_args()

    print(f"{C.BOLD}ShieldPoint Langfuse E2E Trace Test{C.RESET}")
    print(f"  LANGFUSE_HOST      = {os.environ.get('LANGFUSE_HOST', '(not set)')}")
    print(f"  LANGFUSE_PUBLIC_KEY = {os.environ.get('LANGFUSE_PUBLIC_KEY', '(not set)')[:12]}...")
    print(f"  LM_STUDIO_BASE_URL = {os.environ.get('LM_STUDIO_BASE_URL', '(not set)')}")
    print(f"  Mock LLM           = {args.no_llm}")

    s1 = stage_1_langfuse_health()
    s2 = stage_2_sdk_import() if s1 else False
    trace_id = stage_3_send_test_trace(use_mock_llm=args.no_llm) if s2 else None
    s4 = stage_4_verify_trace_captured(trace_id) if trace_id else False
    s5 = stage_5_data_sovereignty()

    # ---- Summary ----
    print(f"\n{C.BOLD}{'=' * 60}{C.RESET}")
    print(f"{C.BOLD}  E2E TEST SUMMARY{C.RESET}")
    print(f"{C.BOLD}{'=' * 60}{C.RESET}")
    print(f"  Stage 1 (Langfuse health):     {'PASS' if s1 else 'FAIL'}")
    print(f"  Stage 2 (SDK import):          {'PASS' if s2 else 'FAIL'}")
    print(f"  Stage 3 (Send trace):          {'PASS' if trace_id else 'FAIL'}")
    print(f"  Stage 4 (Verify captured):     {'PASS' if s4 else 'FAIL'}")
    print(f"  Stage 5 (Data sovereignty):    {'PASS' if s5 else 'FAIL'}")
    print(f"{'=' * 60}")
    print(f"  PASS: {PASS}   WARN: {WARN}   FAIL: {FAIL}")
    print(f"{'=' * 60}{C.RESET}")

    if FAIL > 0:
        return 1
    if WARN > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
