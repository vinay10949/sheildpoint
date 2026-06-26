# ShieldPoint Tool Registry (SP-201)

The standardized interface between the ShieldPoint agent and external systems.
Each tool is a Python function paired with a JSON-Schema descriptor that the
LLM uses to decide which tool to invoke and with what parameters.

Built on top of the existing [`vinay10949/sheildpoint`](https://github.com/vinay10949/sheildpoint)
repo, this package implements the **Epic 1 — Tool-Using Agent** backlog item
requiring a `ToolRegistry` with `register_tool()`, `get_tool()`, `invoke()`,
six built-in claim/policy tools, JSON-Schema validation, Langfuse span
decoration, and unit tests with mocked database responses.

## Quick start

```bash
pip install -e .[test]      # install + test deps
pytest tests/ -v            # 172 tests, ~0.5s
python scripts/demo.py      # end-to-end demo of all 6 tools
```

## The six built-in tools

| Tool | Description |
|------|-------------|
| `claim_lookup`         | Retrieve a claim by ID from the PostgreSQL claims DB. |
| `policy_validate`      | Check policy status / coverage type / effective dates / limits. |
| `payment_authorize`    | Initiate ACH payment with amount validation + duplicate detection. |
| `zkp_prove_policy`     | Generate a Policy Validity Proof (Circom/SnarkJS stub; full impl in SP-202). |
| `zkp_verify_proof`     | Verify a ZKP proof (Groth16 verifier stub, ~10 ms constant time). |
| `claim_update_status`  | Transition a claim through the 8-state machine with guard conditions. |

## Usage

```python
from shieldpoint import build_default_registry, NullSpanRecorder

recorder = NullSpanRecorder()
registry = build_default_registry(span_recorder=recorder)

# Each tool is invoked via the registry — args are validated against the
# JSON Schema before the function runs, and every call is recorded as a
# Langfuse span (with full input / output / latency / error).

claim = registry.invoke("claim_lookup", claim_id="CLM-2026-0001")
# {'claim_id': 'CLM-2026-0001', 'amount': 1250.0, 'status': 'submitted', ...}

policy = registry.invoke(
    "policy_validate", policy_id="HO-2024-001", as_of_date="2026-03-14"
)
# {'policy_id': 'HO-2024-001', 'limit': 250000, 'validation': {'valid': True, ...}}

proof = registry.invoke(
    "zkp_prove_policy",
    claim_id="CLM-2026-0001", policy_id="HO-2024-001",
    claim_amount=1250.0, coverage_limit=250_000,
    peril_covered=True, policy_active=True,
)
# {'proof': 'zkp:...', 'verified': True, 'proof_type': 'groth16-stub', ...}

verify = registry.invoke(
    "zkp_verify_proof",
    proof=proof["proof"], public_signals=proof["public_signals"],
)
# {'verified': True, 'verifier': 'groth16-stub', 'latency_ms': 0.01, ...}

# Idempotent payment with duplicate detection
r1 = registry.invoke(
    "payment_authorize",
    claim_id="CLM-2026-0001", amount=1250.0, payee="Alice Homeowner",
    idempotency_key="user-action-42",
)
# {'status': 'authorized', 'payment_id': 'PMT-CLM-2026-0001-...', ...}

r2 = registry.invoke(
    "payment_authorize",
    claim_id="CLM-2026-0001", amount=1250.0, payee="Alice Homeowner",
    idempotency_key="user-action-42",
)
# {'status': 'duplicate_detected', 'duplicate_of': 'PMT-...', ...}

# State machine transitions with guard conditions
registry.invoke(
    "claim_update_status",
    claim_id="CLM-2026-0001", new_status="validating",
    context={},
)
# {'status': 'ok', 'previous_status': 'claim_received', 'new_status': 'validating', ...}

# Inspect the recorded spans
for span in recorder.spans:
    print(f"{span.name:25s} {span.status:18s} {span.latency_ms:.2f}ms")
```

## Architecture

```
shieldpoint/
├── __init__.py              # Public API surface
├── tool_registry.py         # ToolRegistry + Tool + exceptions
├── langfuse_span.py         # SpanRecorder protocol + Null / Langfuse impls
├── db.py                    # Repository protocols + in-memory impls
├── state_machine.py         # 8-state, 9-transition claim state machine
├── zkp.py                   # Circom/SnarkJS prover + Groth16 verifier stubs
└── tools/
    └── __init__.py          # The 6 built-in tools + build_default_registry()

tests/                       # 172 unit + integration tests
├── conftest.py              # Fresh per-test fixtures + mock repos
├── test_tool_registry.py    # 24 tests — register / get / invoke / validation
├── test_claim_lookup.py     # 10 tests
├── test_policy_validate.py  # 13 tests
├── test_payment_authorize.py# 20 tests — including duplicate detection
├── test_zkp_tools.py        # 19 tests — prover + verifier + roundtrip
├── test_claim_update_status.py # 40 tests — state machine + guards
├── test_db.py               # 22 tests — in-memory repos
├── test_langfuse_span.py    # 14 tests — span recording
└── test_integration.py      # 10 tests — end-to-end flow
```

## Claim state machine

The 8 states and 9 transitions are defined in Section 5 of the ShieldPoint
Claims Automation Implementation Plan v2.0:

```
CLAIM_RECEIVED ──────────────► VALIDATING
                                  │
                                  │ (guard: all required fields present)
                                  ▼
                            ZKP_POLICY_PROOF ──────► ESCALATING
                                  │  ▲                  ▲
                                  │  │                  │
              (guard: proof_verified + confidence≥0.85) │ (proof failed)
                                  │  │                  │
                                  ▼  │ (guard: human_approval)
                            CLASSIFYING                 │
                                  │                     │
              (guard: severity + fraud_score computed)  │
                                  ▼                     │
                       ZKP_COMPLIANCE_PROOF ────────────┘
                                  │
              (guard: compliance_proved + low_risk + confidence≥0.85)
                                  ▼
                              APPROVED ──────────► PAID_OUT
                                  ▲              (guard: payment_authorized +
                                  │                       bank_details_verified)
                          (from ESCALATING)
```

Each transition is guarded by an explicit condition (see
`shieldpoint/state_machine.py`). Guard failures return a structured error
dict (not an exception) so the agent can feed the reason back to the LLM.

## PostgreSQL production wiring

The `db.py` module defines three repository protocols:
`ClaimsRepository`, `PolicyRepository`, `PaymentLedgerRepository`.
The in-memory implementations (`InMemoryClaimsRepository`, etc.) are seeded
with the ShieldPoint demo dataset and used for unit tests.

For production, implement the same protocols against PostgreSQL. The
exact SQL schema is documented in the docstring of `db.py`. Pass the
Postgres-backed repos to `build_default_registry()`:

```python
from shieldpoint import build_default_registry
from myapp.db import PostgresClaimsRepository, PostgresPolicyRepository, ...

registry = build_default_registry(
    claims_repo=PostgresClaimsRepository(dsn="..."),
    policy_repo=PostgresPolicyRepository(dsn="..."),
    payment_repo=PostgresPaymentLedgerRepository(dsn="..."),
    span_recorder=LangfuseSpanRecorder(),
)
```

## Langfuse span recording

Every `registry.invoke()` call is wrapped in a span via the injected
`SpanRecorder`. Two implementations are shipped:

- `NullSpanRecorder` (default) — captures spans in memory; ideal for tests.
- `LangfuseSpanRecorder` — ships spans to a live Langfuse server. Silently
  degrades to local-only capture if the `langfuse` SDK is missing or
  `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` env vars are unset.

Each span captures: tool name, input kwargs, output, error (if any), and
wall-clock latency in milliseconds.

## ZKP stubs vs. production (SP-202)

The `zkp_prove_policy` and `zkp_verify_proof` tools are **stubs** that
produce deterministic, hash-based proofs. They are sufficient for wiring up
the agent stack and running end-to-end tests, but are NOT cryptographically
secure. The production implementation (SP-202) will replace the function
bodies with subprocess calls to SnarkJS:

```python
# SP-202 will replace the body of verify_policy_validity_proof with:
subprocess.run(
    ["snarkjs", "groth16", "verify", "verification_key.json",
     "public.json", "proof.json"],
    check=True, capture_output=True,
)
```

The function signatures and return shapes are stable, so the swap is a
drop-in.

## Acceptance criteria mapping

| Acceptance criterion | Implementation |
|----------------------|----------------|
| `ToolRegistry` with `register_tool()`, `get_tool()`, `invoke()` | `shieldpoint/tool_registry.py` |
| Each tool has name, description, JSON schema, Python impl | `shieldpoint/tools/__init__.py` (`TOOL_SCHEMAS`, `TOOL_DESCRIPTIONS`, 6 functions) |
| `claim_lookup` from PostgreSQL | `claim_lookup` tool + `ClaimsRepository` protocol + `InMemoryClaimsRepository` (Postgres impl documented in `db.py`) |
| `policy_validate` checks status / coverage / dates / limits | `policy_validate` tool returns `validation` dict with `is_active`, `is_in_force`, `coverage_ok`, `issues` |
| `payment_authorize` with amount validation + duplicate detection | `payment_authorize` tool with idempotency-key + (claim, amount, payee) dedup |
| `zkp_prove_policy` calls Circom/SnarkJS prover | `zkp_prove_policy` tool (stub in `zkp.py`; SP-202 replaces with SnarkJS subprocess) |
| `zkp_verify_proof` calls Groth16 verifier (10 ms constant) | `zkp_verify_proof` tool (stub; test asserts <10 ms latency) |
| `claim_update_status` with state machine guard conditions | `claim_update_status` tool + `ClaimStateMachine` with 8 states, 9 transitions, guard functions |
| All tool invocations logged as Langfuse spans | `SpanRecorder` protocol; `invoke()` opens span around every call |
| Schema validation before execution | `Tool.validate_kwargs()` via `jsonschema.Draft7Validator` |
| Structured error messages for invalid inputs | `ToolValidationError` carries `details` dict with `json_path`, `schema_path`, `validator`, etc. |
| Unit tests with mocked DB responses | 172 tests; `MockClaimsRepository` / `MockPolicyRepository` / `MockPaymentLedgerRepository` in `conftest.py` |

## License

Proprietary — ShieldPoint Insurance. For internal use only.
