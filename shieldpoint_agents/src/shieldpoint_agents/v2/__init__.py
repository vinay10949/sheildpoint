"""
ShieldPoint Agent V2 — Five-Agent State Machine Roles
=====================================================

This package implements the five specialist agents that own individual
states in the :mod:`state_machine_engine` lifecycle:

- :class:`IntakeAgent`        — owns ``CLAIM_RECEIVED``
- :class:`ValidatorAgent`     — owns ``VALIDATING``
- :class:`ClassifierAgent`    — owns ``CLASSIFYING``
- :class:`EscalationAgent`    — owns ``ESCALATING`` (HITL)
- :class:`PayoutAgent`        — owns ``APPROVED`` → ``PAID_OUT``

Each agent is a thin, deterministic Python class. It accepts a claim dict
plus a context dict, performs its state's work, and returns an updated
context dict that the engine's guard functions consume. The agents do NOT
call the state machine directly — they are pure functions of (claim,
context) → context, which keeps them trivially unit-testable and lets the
engine own all state transitions.

ZKP gates are owned by the ZKP prover module (``zkp_circuit/compliance_prover.py``
for the compliance gate) rather than by an agent class — the gate is a
cryptographic primitive, not an LLM-driven step.

LLM integration
---------------
:class:`ClassifierAgent` is the only agent that calls the LLM (Qwen 3.6 via
LM Studio, following the repo's existing pattern). It does so via the
``llm_client`` injection point — tests pass a :class:`FakeLLMClient` to
make classification deterministic; production passes a real
``openai.OpenAI`` client configured against ``LM_STUDIO_BASE_URL``.

Langfuse spans
--------------
All agents emit Langfuse spans around their work via the same
``langfuse_wrapper.py`` used by the rest of the repo. Spans tag the
``claim_id``, ``agent_id``, and structured output so reviewers can trace
any decision back to the exact prompt + reasoning.
"""

from .agents import (
    ClassifierAgent,
    EscalationAgent,
    FakeLLMClient,
    IntakeAgent,
    PayoutAgent,
    ValidatorAgent,
)
from .silos import (
    BillingSilo,
    DocumentManagementSilo,
    InMemorySiloStore,
    PolicyAdministrationSilo,
    SiloRecord,
    UnderwritingSilo,
)
# SP-405: Payout subsystem exports
from .payout import (
    ACHProvider,
    ACHResult,
    StubACHProvider,
    BankVerificationService,
    PaymentLedger,
    PaymentRecord,
    InMemoryPaymentLedger,
    ReceiptGenerator,
    ReceiptResult,
    NotificationService,
    NotificationResult,
    StubNotificationService,
    AuditRecordAssembler,
    AuditRecord,
)


__all__ = [
    "IntakeAgent",
    "ValidatorAgent",
    "ClassifierAgent",
    "EscalationAgent",
    "PayoutAgent",
    "FakeLLMClient",
    "InMemorySiloStore",
    "SiloRecord",
    "PolicyAdministrationSilo",
    "BillingSilo",
    "UnderwritingSilo",
    "DocumentManagementSilo",
    # SP-405: Payout subsystem
    "ACHProvider",
    "ACHResult",
    "StubACHProvider",
    "BankVerificationService",
    "PaymentLedger",
    "PaymentRecord",
    "InMemoryPaymentLedger",
    "ReceiptGenerator",
    "ReceiptResult",
    "NotificationService",
    "NotificationResult",
    "StubNotificationService",
    "AuditRecordAssembler",
    "AuditRecord",
]
