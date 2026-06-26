"""
Specialist agents for the ManagerAgent (SHLD-15).

Three specialist subclasses of :class:`Agent`, each focused on a single
dimension of the claim assessment:

- :class:`ClaimsAgent` — policy validation + claim-history + claim-lookup.
  Authoritative on whether the policy covers the loss event.
- :class:`FinancialAgent` — coverage limits, deductibles, payment
  authorisation, ZKP proof generation. Authoritative on money questions.
- :class:`SentimentAgent` — sentiment & urgency analysis of the claim
  description (e.g. hostile tone, fraud markers, attorney threats).
  Advisory — never authoritative alone.

Each specialist:

- Inherits the base ReAct loop, fallback engine, confidence scorer, and
  HITL escalator from :class:`Agent`.
- Pre-registers the subset of tools from ``claims_tools.py`` relevant to
  its dimension.
- Carries a customised system prompt that biases its reasoning toward
  its dimension.
- Accepts the standard ``llm_client`` injection point so tests can
  drive it with :class:`FakeLMClient`.

The specialists do not need to be invoked directly — the ManagerAgent
constructs and orchestrates them. But they remain independently runnable
(a useful property for debugging and for unit-testing each dimension in
isolation).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .agent import _SYSTEM_PROMPT as _BASE_SYSTEM_PROMPT
from .claims_tools import get_tool_schemas
from .claims_tools import (
    check_claim_history,
    claim_lookup,
    generate_zkp_proof,
    process_payment,
    validate_policy,
)
from .config import AgentConfig
from .confidence import ConfidenceScorer
from .escalation import HITLEscalator
from .fallback import FallbackEngine
from .schemas import AgentRunResult
from .tools import ToolRegistry
from .tracer import LangfuseTracer
from .agent import Agent

logger = logging.getLogger("shieldpoint_agents.specialists")


# ---------------------------------------------------------------------------
# Specialist system prompts — each biases toward one dimension.
# ---------------------------------------------------------------------------
_CLAIMS_PROMPT = """\
You are the ClaimsAgent in the ShieldPoint multi-agent system.

Your dimension: **policy & claim-record validation**. You answer:
- Does the policy cover the claimed peril?
- Is the policy active and within term?
- Has the claimant filed prior claims (frequency / fraud signals)?

Available tools:
{tools_block}

For each step, respond with ONLY a JSON object of this exact shape:
{{
  "thought": "<your reasoning>",
  "action": "<tool name or 'FINAL_ANSWER'>",
  "action_input": {{ <tool args, or the final decision> }}
}}

When you have enough information, set action="FINAL_ANSWER" with:
{{
  "decision": "approve" | "deny" | "route_to_manual_review",
  "reasoning": "<focused on policy coverage & history>",
  "confidence": <float in [0,1]>,
  "evidence": ["peril=X is covered by policy Y", ...]
}}

IMPORTANT:
- Always validate the policy before deciding.
- Always check claim history for frequent claimants.
- If the peril is explicitly excluded, deny.
- If you cannot confirm coverage, route to manual review.
"""


_FINANCIAL_PROMPT = """\
You are the FinancialAgent in the ShieldPoint multi-agent system.

Your dimension: **financial coverage**. You answer:
- Is the claim amount within the policy coverage limit?
- Is the amount above the deductible (so payment is warranted)?
- Can a payment be authorised (and a ZKP proof generated)?

Available tools:
{tools_block}

For each step, respond with ONLY a JSON object of this exact shape:
{{
  "thought": "<your reasoning>",
  "action": "<tool name or 'FINAL_ANSWER'>",
  "action_input": {{ <tool args, or the final decision> }}
}}

When you have enough information, set action="FINAL_ANSWER" with:
{{
  "decision": "approve" | "deny" | "route_to_manual_review",
  "reasoning": "<focused on limits, deductibles, payment viability>",
  "confidence": <float in [0,1]>,
  "evidence": ["amount=1250.00 <= limit=25000.00", ...]
}}

IMPORTANT:
- Always validate the policy to read its limit and deductible.
- Deny if claim amount > policy limit.
- Deny if claim amount < deductible (no payment warranted).
- For large claims, recommend a ZKP proof be generated.
"""


_SENTIMENT_PROMPT = """\
You are the SentimentAgent in the ShieldPoint multi-agent system.

Your dimension: **sentiment, urgency, and fraud-signal analysis** of the
claim description. You do NOT make coverage decisions yourself — your
FINAL_ANSWER reflects whether the *sentiment & signals* suggest the
claim should be approved (calm, cooperative, no red flags), denied
(hostile, fraud markers, attorney threats), or routed to manual review
(mixed signals, mild urgency).

You have access to claim_lookup so you can read the description and
metadata. You do NOT need to validate the policy or authorise payments —
leave that to the other specialists.

Available tools:
{tools_block}

For each step, respond with ONLY a JSON object of this exact shape:
{{
  "thought": "<your reasoning>",
  "action": "<tool name or 'FINAL_ANSWER'>",
  "action_input": {{ <tool args, or the final decision> }}
}}

When you have enough information, set action="FINAL_ANSWER" with:
{{
  "decision": "approve" | "deny" | "route_to_manual_review",
  "reasoning": "<focused on tone, urgency, fraud markers>",
  "confidence": <float in [0,1]>,
  "evidence": ["tone=cooperative", "no fraud markers detected", ...]
}}

IMPORTANT:
- Look for explicit fraud markers: "fraud", "intentional", "misrepresentation".
- Look for urgency markers: "injury", "attorney", "litigation", "urgent".
- Sentiment is advisory — confidence should rarely exceed 0.85.
"""


# ---------------------------------------------------------------------------
# Helper — build a specialist agent with a curated tool subset.
# ---------------------------------------------------------------------------
def _build_specialist_registry(tool_names: list[str]) -> ToolRegistry:
    """Build a ToolRegistry containing only the named tools from claims_tools."""
    schemas = get_tool_schemas()
    name_to_fn = {
        "claim_lookup": claim_lookup,
        "validate_policy": validate_policy,
        "check_claim_history": check_claim_history,
        "process_payment": process_payment,
        "generate_zkp_proof": generate_zkp_proof,
    }
    registry = ToolRegistry()
    for name in tool_names:
        fn = name_to_fn.get(name)
        if fn is None:
            logger.warning("Unknown tool %r requested for specialist", name)
            continue
        registry.register(fn, name=name, schema=schemas[name])
    return registry


# ---------------------------------------------------------------------------
# ClaimsAgent
# ---------------------------------------------------------------------------
class ClaimsAgent(Agent):
    """Specialist — policy & claim-record validation.

    Authoritative on whether the policy covers the loss event. Tools:
    ``validate_policy``, ``check_claim_history``, ``claim_lookup``.

    SP-301 enhancements:
    - :meth:`extract_and_validate` — runs the full extraction / normalisation
      / validation / ZKP-proof pipeline on a raw claim. Returns an
      :class:`ExtractionEnvelope` (see ``claims_extraction.py``).
    """

    def __init__(
        self,
        *,
        config: Optional[AgentConfig] = None,
        llm_client: Any = None,
        tracer: Optional[LangfuseTracer] = None,
        fallback: Optional[FallbackEngine] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        hitl_escalator: Optional[HITLEscalator] = None,
        extraction_pipeline: Any = None,
    ) -> None:
        registry = _build_specialist_registry([
            "claim_lookup",
            "validate_policy",
            "check_claim_history",
        ])
        super().__init__(
            name="ClaimsAgent",
            tools=registry,
            tracer=tracer or LangfuseTracer(agent_name="ClaimsAgent"),
            fallback=fallback or FallbackEngine(),
            config=config or AgentConfig.from_env(),
            llm_client=llm_client,
            system_prompt=_CLAIMS_PROMPT,
            confidence_scorer=confidence_scorer,
            hitl_escalator=hitl_escalator,
        )
        # SP-301: lazy-init the extraction pipeline on first use so the
        # ClaimsAgent constructor stays cheap when extract_and_validate
        # isn't needed.
        self._extraction_pipeline = extraction_pipeline

    def extract_and_validate(
        self,
        raw_claim: Any,
        *,
        claim_id: Optional[str] = None,
        policy_coverage_limit: Optional[float] = None,
        policy_id_numeric: Optional[int] = None,
        policy_salt: Optional[int] = None,
    ) -> Any:
        """SP-301 entry point — extract, normalise, validate, generate ZKP proof.

        See :class:`ClaimsExtractionPipeline.run` for parameter docs.
        Returns an :class:`ExtractionEnvelope`.

        If the pipeline hasn't been pre-injected (the common case in
        production), one is constructed lazily sharing this agent's
        ``llm_client`` and ``tracer``.
        """
        if self._extraction_pipeline is None:
            from .claims_extraction import ClaimsExtractionPipeline
            self._extraction_pipeline = ClaimsExtractionPipeline(
                config=self.config,
                llm_client=self._llm_client,
                tracer=self.tracer,
            )
        return self._extraction_pipeline.run(
            raw_claim,
            claim_id=claim_id,
            policy_coverage_limit=policy_coverage_limit,
            policy_id_numeric=policy_id_numeric,
            policy_salt=policy_salt,
        )


# ---------------------------------------------------------------------------
# FinancialAgent
# ---------------------------------------------------------------------------
class FinancialAgent(Agent):
    """Specialist — financial coverage, limits, payment authorisation.

    Authoritative on money questions. Tools: ``validate_policy``,
    ``process_payment``, ``generate_zkp_proof``.

    SP-302 enhancements:
    - :meth:`assess_payment` — runs the full payment calculation engine
      (deductible calculator + co-pay + duplicate detection + ZKP proof
      verification) and emits a :class:`PaymentAuthorizationRecord`.
    """

    def __init__(
        self,
        *,
        config: Optional[AgentConfig] = None,
        llm_client: Any = None,
        tracer: Optional[LangfuseTracer] = None,
        fallback: Optional[FallbackEngine] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        hitl_escalator: Optional[HITLEscalator] = None,
        assessment_engine: Any = None,
    ) -> None:
        registry = _build_specialist_registry([
            "validate_policy",
            "process_payment",
            "generate_zkp_proof",
        ])
        super().__init__(
            name="FinancialAgent",
            tools=registry,
            tracer=tracer or LangfuseTracer(agent_name="FinancialAgent"),
            fallback=fallback or FallbackEngine(),
            config=config or AgentConfig.from_env(),
            llm_client=llm_client,
            system_prompt=_FINANCIAL_PROMPT,
            confidence_scorer=confidence_scorer,
            hitl_escalator=hitl_escalator,
        )
        # SP-302: lazy-init the assessment engine on first use.
        self._assessment_engine = assessment_engine

    def assess_payment(self, **kwargs: Any) -> Any:
        """SP-302 entry point — verify ZKP, check duplicates, calculate payment.

        See :meth:`FinancialAssessmentEngine.assess` for parameter docs.
        Returns a :class:`PaymentAuthorizationRecord`.
        """
        if self._assessment_engine is None:
            from .financial_engine import FinancialAssessmentEngine
            self._assessment_engine = FinancialAssessmentEngine(
                config=self.config,
                tracer=self.tracer,
            )
        return self._assessment_engine.assess(**kwargs)


# ---------------------------------------------------------------------------
# SentimentAgent
# ---------------------------------------------------------------------------
class SentimentAgent(Agent):
    """Specialist — sentiment, urgency, and fraud-signal analysis.

    Advisory only. Tool: ``claim_lookup`` (read-only).

    SP-303 enhancements:
    - :meth:`analyze_sentiment` — runs the multi-dimensional sentiment
      analysis (urgency / emotional_state / veracity) and returns a
      :class:`SentimentAssessment`.
    """

    def __init__(
        self,
        *,
        config: Optional[AgentConfig] = None,
        llm_client: Any = None,
        tracer: Optional[LangfuseTracer] = None,
        fallback: Optional[FallbackEngine] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        hitl_escalator: Optional[HITLEscalator] = None,
        sentiment_engine: Any = None,
    ) -> None:
        registry = _build_specialist_registry(["claim_lookup"])
        super().__init__(
            name="SentimentAgent",
            tools=registry,
            tracer=tracer or LangfuseTracer(agent_name="SentimentAgent"),
            fallback=fallback or FallbackEngine(),
            config=config or AgentConfig.from_env(),
            llm_client=llm_client,
            system_prompt=_SENTIMENT_PROMPT,
            confidence_scorer=confidence_scorer,
            hitl_escalator=hitl_escalator,
        )
        # SP-303: lazy-init the sentiment engine on first use.
        self._sentiment_engine = sentiment_engine

    def analyze_sentiment(self, text: str, *, claim_id: Optional[str] = None) -> Any:
        """SP-303 entry point — multi-dimensional sentiment analysis.

        See :class:`SentimentAnalysisEngine.analyze` for parameter docs.
        Returns a :class:`SentimentAssessment`.
        """
        if self._sentiment_engine is None:
            from .sentiment import SentimentAnalysisEngine
            self._sentiment_engine = SentimentAnalysisEngine(
                config=self.config,
                llm_client=self._llm_client,
                tracer=self.tracer,
            )
        return self._sentiment_engine.analyze(text, claim_id=claim_id)


# ---------------------------------------------------------------------------
# Registry — build all three specialists with a shared LLM client
# ---------------------------------------------------------------------------
def build_specialists(
    *,
    config: Optional[AgentConfig] = None,
    llm_client_factory: Optional[Any] = None,
    fallback: Optional[FallbackEngine] = None,
    confidence_scorer: Optional[ConfidenceScorer] = None,
    hitl_escalator: Optional[HITLEscalator] = None,
) -> dict[str, Agent]:
    """Build the standard trio of specialist agents.

    ``llm_client_factory`` is an optional callable that returns a fresh
    LLM client for each specialist (so each has its own response queue
    in tests). If ``None``, the specialists will lazily construct their
    own client from ``config`` on first use.
    """
    agents: dict[str, Agent] = {}
    for cls in (ClaimsAgent, FinancialAgent, SentimentAgent):
        kwargs: dict[str, Any] = {}
        if config is not None:
            kwargs["config"] = config
        if fallback is not None:
            kwargs["fallback"] = fallback
        if confidence_scorer is not None:
            kwargs["confidence_scorer"] = confidence_scorer
        if hitl_escalator is not None:
            kwargs["hitl_escalator"] = hitl_escalator
        if llm_client_factory is not None:
            kwargs["llm_client"] = llm_client_factory()
        agents[cls.__name__] = cls(**kwargs)
    return agents
