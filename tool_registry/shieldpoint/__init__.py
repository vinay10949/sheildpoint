"""
ShieldPoint Tool Registry (SP-201 / Epic 1 — Tool-Using Agent)

This package implements the standardized interface between the ShieldPoint
agent and external systems (PostgreSQL claims DB, policy DB, ACH payment
gateway, Circom/SnarkJS prover, Groth16 verifier). Each tool is a plain
Python function paired with a JSON-Schema descriptor that the LLM uses to
decide which tool to invoke and with what parameters.

Public API
----------
- :class:`ToolRegistry`         — register / lookup / invoke tools with schema validation + Langfuse spans
- :class:`Tool`                 — a single registered tool (function + JSON-Schema + metadata)
- :class:`ToolNotFoundError`    — raised when ``invoke()`` is called on an unknown tool
- :class:`ToolValidationError`  — raised when kwargs fail JSON-Schema validation
- :class:`ToolInvocationError`  — wraps any exception raised by the underlying function
- :func:`build_default_registry` — returns a ToolRegistry pre-loaded with the six
  claim/policy tools required by SP-201.

The six built-in tools
----------------------
- ``claim_lookup``         — retrieve a claim by ID from PostgreSQL
- ``policy_validate``      — check policy status / coverage / effective dates / limits
- ``payment_authorize``    — initiate ACH payment with amount validation + duplicate detection
- ``zkp_prove_policy``     — call Circom/SnarkJS prover to generate a Policy Validity Proof (stub)
- ``zkp_verify_proof``     — call Groth16 verifier in ~10 ms constant time (stub)
- ``claim_update_status``  — transition a claim through the 8-state machine with guard conditions
"""

from __future__ import annotations

from .tool_registry import (
    Tool,
    ToolInvocationError,
    ToolNotFoundError,
    ToolRegistry,
    ToolValidationError,
)
from .langfuse_span import LangfuseSpanRecorder, NullSpanRecorder
from .db import (
    ClaimsRepository,
    PolicyRepository,
    PaymentLedgerRepository,
    InMemoryClaimsRepository,
    InMemoryPolicyRepository,
    InMemoryPaymentLedgerRepository,
)
from .state_machine import (
    ClaimState,
    ClaimStateMachine,
    InvalidStateTransitionError,
    GuardConditionFailedError,
)
from .tools import (
    claim_lookup,
    policy_validate,
    payment_authorize,
    zkp_prove_policy,
    zkp_verify_proof,
    claim_update_status,
    build_default_registry,
)

__all__ = [
    # Core registry
    "ToolRegistry",
    "Tool",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolInvocationError",
    # Observability
    "LangfuseSpanRecorder",
    "NullSpanRecorder",
    # Repositories
    "ClaimsRepository",
    "PolicyRepository",
    "PaymentLedgerRepository",
    "InMemoryClaimsRepository",
    "InMemoryPolicyRepository",
    "InMemoryPaymentLedgerRepository",
    # State machine
    "ClaimState",
    "ClaimStateMachine",
    "InvalidStateTransitionError",
    "GuardConditionFailedError",
    # Built-in tools
    "claim_lookup",
    "policy_validate",
    "payment_authorize",
    "zkp_prove_policy",
    "zkp_verify_proof",
    "claim_update_status",
    "build_default_registry",
]

__version__ = "0.1.0"
