"""
ShieldPoint Agent Framework
===========================

Base agent framework for ShieldPoint claims automation. Provides:

- :class:`Agent` — base class with a Think/Plan/Act (ReAct) loop, structured
  output parsing, confidence scoring, HITL escalation, and graceful fallback
  to rule-based processing when the LLM call fails or times out (>10s).
- :class:`ToolRegistry` — registers Python callables with JSON-schema
  descriptors, validates arguments on every invocation, and logs each call
  to Langfuse.
- :class:`LangfuseTracer` — per-instance façade over the existing
  ``agent_framework.observability.ShieldPointTracer`` singleton. Decorates
  every LLM call and tool invocation so prompts, completions, latency, and
  token counts are captured to the self-hosted Langfuse stack.
- :class:`FallbackEngine` — deterministic claim-processing rules used when
  the LLM is unavailable.
- :class:`ConfidenceScorer` — multi-signal confidence scorer combining
  self-assessment, consistency, evidence grounding, and tool coverage.
- :class:`HITLEscalator` — handles HITL escalation when confidence drops
  below configurable thresholds.

Package layout
--------------

::

    shieldpoint_agents/
    ├── __init__.py          # public API surface (this file)
    ├── _bootstrap.py        # sys.path setup so `agent_framework.*` imports work
    ├── config.py            # AgentConfig dataclass, reads env vars at call time
    ├── tracer.py            # LangfuseTracer class
    ├── schemas.py           # Pydantic models for ReAct structured output
    ├── tools.py             # Tool + ToolRegistry
    ├── fallback.py          # FallbackEngine
    ├── confidence.py        # ConfidenceScorer (SHLD-14)
    ├── escalation.py        # HITLEscalator (SHLD-14)
    ├── claims_tools.py      # Claims-specific tools (SHLD-14)
    ├── agent.py             # Agent base class with ReAct loop
    ├── _lmstudio.py         # OpenAI-compatible client factory for LM Studio
    ├── api.py               # FastAPI demo: /health + /run
    └── example.py           # CLI demo: `python -m shieldpoint_agents.example`

Environment variables
---------------------

The package reads these at runtime (so monkeypatching works in tests):

- ``LM_STUDIO_BASE_URL``  — default ``http://localhost:1234/v1``
- ``LM_STUDIO_API_KEY``   — default ``lm-studio``
- ``QWEN_MODEL_ID``       — default ``qwen3.6-35b-a3b-q4_k_m``
- ``LLM_TIMEOUT_SEC``     — default ``10`` (per AC: >10s triggers fallback)
- ``MAX_REACT_ITERATIONS`` — default ``10`` (SHLD-14)
- ``HITL_CONFIDENCE_THRESHOLD`` — default ``0.85`` (SHLD-14)
- ``FALLBACK_CONFIDENCE_THRESHOLD`` — default ``0.50`` (SHLD-14)
- ``LANGFUSE_HOST``       — default ``http://localhost:3000``
- ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` — required for tracing
- ``LANGFUSE_ENABLED``    — default ``true``; set ``false`` to no-op traces

Usage
-----

::

    from shieldpoint_agents import (
        Agent, ToolRegistry, LangfuseTracer, FallbackEngine,
        ConfidenceScorer, HITLEscalator,
    )

    registry = ToolRegistry()
    registry.register(validate_policy, schema={
        "type": "object",
        "properties": {"policy_id": {"type": "string"}},
        "required": ["policy_id"],
    })

    agent = Agent(
        name="claim-classifier",
        tools=registry,
        tracer=LangfuseTracer(),
        fallback=FallbackEngine(),
        confidence_scorer=ConfidenceScorer(),
        hitl_escalator=HITLEscalator(),
    )

    decision = agent.run(claim={
        "claim_id": "CLM-2026-0001",
        "amount": 1250.00,
        "description": "Wind damage to roof shingles.",
        "policy_id": "HO-2024-001",
    })
    print(decision)
"""

from __future__ import annotations

from ._bootstrap import ensure_repo_root_on_path

# Side-effect import: makes `agent_framework.observability.ShieldPointTracer`
# importable when this package is installed editable or site-installed.
ensure_repo_root_on_path()

from .config import AgentConfig
from .tracer import LangfuseTracer
from .tools import Tool, ToolRegistry
from .fallback import FallbackEngine, FallbackResult
from .schemas import ReActStep, ClaimDecision, AgentRunResult
from .confidence import ConfidenceScorer, ConfidenceReport, ConfidenceWeights
from .escalation import HITLEscalator, EscalationRecord
from .agent import Agent, AgentError

# ---- SHLD-15: Multi-agent orchestration (ManagerAgent) ----
from .manager_schemas import (
    AgentInvocationRecord,
    ConflictRecord,
    EpisodicMemoryEntry,
    ManagerRunResult,
    OrchestrationPlan,
    OrchestrationStage,
)
from .memory import (
    EpisodicMemoryStore,
    InMemoryEpisodicMemory,
    PostgresEpisodicMemory,
    build_episodic_memory,
    make_entry_from_result,
)
from .conflict import (
    ConflictDetector,
    ConflictResolution,
    ConflictResolver,
    DEFAULT_PRIORITY_MAP,
)
from .specialists import (
    ClaimsAgent,
    FinancialAgent,
    SentimentAgent,
    build_specialists,
)
from .manager import ManagerAgent

# ---- SP-301: ClaimsAgent data extraction & formatting ----
from .claims_extraction import (
    AddressNormalizer,
    ClaimsExtractionPipeline,
    CompletenessValidator,
    CurrencyNormalizer,
    DateNormalizer,
    ExtractionEnvelope,
    LLMFieldExtractor,
    make_standard_claim,
)

# ---- SP-302: FinancialAgent payment assessment engine ----
from .financial_engine import (
    DeductibleCalculator,
    DuplicatePaymentDetector,
    FinancialAssessmentEngine,
    PaymentAuthorizationRecord,
    PaymentCalculator,
    PriorClaim,
    ZKPCrossAgentVerifier,
    build_financial_scenarios,
)

# ---- SP-303: SentimentAgent multi-dimensional analysis ----
from .sentiment import (
    LLMSentimentAnalyzer,
    LABELED_DATASET,
    RuleBasedSentimentAnalyzer,
    SentimentAnalysisEngine,
    SentimentAssessment,
    SentimentOutputParser,
)

__all__ = [
    # ---- base ----
    "Agent",
    "AgentConfig",
    "AgentError",
    "AgentRunResult",
    "ClaimDecision",
    "ConfidenceReport",
    "ConfidenceScorer",
    "ConfidenceWeights",
    "EscalationRecord",
    "FallbackEngine",
    "FallbackResult",
    "HITLEscalator",
    "LangfuseTracer",
    "ReActStep",
    "Tool",
    "ToolRegistry",
    # ---- SHLD-15: multi-agent orchestration ----
    "AgentInvocationRecord",
    "ClaimsAgent",
    "ConflictDetector",
    "ConflictRecord",
    "ConflictResolution",
    "ConflictResolver",
    "DEFAULT_PRIORITY_MAP",
    "EpisodicMemoryEntry",
    "EpisodicMemoryStore",
    "FinancialAgent",
    "InMemoryEpisodicMemory",
    "ManagerAgent",
    "ManagerRunResult",
    "OrchestrationPlan",
    "OrchestrationStage",
    "PostgresEpisodicMemory",
    "SentimentAgent",
    "build_episodic_memory",
    "build_specialists",
    "make_entry_from_result",
    # ---- SP-301: ClaimsAgent extraction pipeline ----
    "AddressNormalizer",
    "ClaimsExtractionPipeline",
    "CompletenessValidator",
    "CurrencyNormalizer",
    "DateNormalizer",
    "ExtractionEnvelope",
    "LLMFieldExtractor",
    "make_standard_claim",
    # ---- SP-302: FinancialAgent payment assessment ----
    "DeductibleCalculator",
    "DuplicatePaymentDetector",
    "FinancialAssessmentEngine",
    "PaymentAuthorizationRecord",
    "PaymentCalculator",
    "PriorClaim",
    "ZKPCrossAgentVerifier",
    "build_financial_scenarios",
    # ---- SP-303: SentimentAgent multi-dimensional analysis ----
    "LABELED_DATASET",
    "LLMSentimentAnalyzer",
    "RuleBasedSentimentAnalyzer",
    "SentimentAnalysisEngine",
    "SentimentAssessment",
    "SentimentOutputParser",
]

__version__ = "0.3.0"
