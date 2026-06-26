"""
ManagerAgent — multi-agent orchestrator (SHLD-15).

The :class:`ManagerAgent` extends the base :class:`Agent` class to
orchestrate three specialist agents (:class:`ClaimsAgent`,
:class:`FinancialAgent`, :class:`SentimentAgent`) for multi-claim
processing.

Responsibilities
----------------

1. **Routing** — classify the claim into a high-level type (property,
   liability, auto_collision, theft, fraud_suspected) and produce an
   :class:`OrchestrationPlan` describing which specialists to invoke,
   in what sequence, and in what execution mode (sequential or
   parallel).

2. **Invocation** — execute the plan stage-by-stage. Inside a stage,
   specialists can run sequentially (each sees prior stage results via
   episodic memory) or in parallel (no intra-stage data sharing).

3. **Synthesis** — combine specialist outputs into a single unified
   :class:`ClaimDecision`. When specialists agree, the manager averages
   confidence and merges evidence. When they disagree, the
   :class:`ConflictResolver` applies the configured strategy.

4. **Episodic memory** — every specialist output is appended to the
   :class:`EpisodicMemoryStore`. On follow-up interactions with the
   same claim, prior episodes are recalled and surfaced to the
   specialists as additional context.

5. **Tracing** — every orchestration decision, specialist invocation,
   and conflict resolution is logged as a linked span within a single
   Langfuse trace tree. The manager opens the top-level trace;
   specialist ``Agent.run`` calls open nested spans that auto-attach
   via OpenTelemetry context vars.

Routing logic (claim type → plan)
---------------------------------

- **property** (homeowners, wind/hail/fire/theft) → all three
  specialists, parallel. Property claims need coverage + financial +
  sentiment (potential fraud) checks together.
- **liability** (auto with injury, attorney mentions) → ClaimsAgent
  + SentimentAgent first (parallel), then FinancialAgent. The
  financial agent runs last because it may need to consider the
  liability findings.
- **auto_collision** (no injury) → ClaimsAgent + FinancialAgent only.
  Skip sentiment (no fraud signal expected for routine collisions).
- **theft** → all three, sequential: SentimentAgent first (fraud
  signals), then ClaimsAgent, then FinancialAgent.
- **fraud_suspected** (description contains fraud markers) → all three,
  sequential, with SentimentAgent first to flag urgency.
- **unknown** → all three, parallel (conservative default).

The routing rules are intentionally simple and inspectable — they live
in :meth:`ManagerAgent._classify_claim_type` and
:meth:`ManagerAgent._build_plan_for_type` and can be overridden in
subclasses for product-line-specific tuning.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from .agent import Agent
from .config import AgentConfig
from .confidence import ConfidenceScorer
from .conflict import ConflictResolver
from .escalation import HITLEscalator
from .fallback import FallbackEngine
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
    make_entry_from_result,
)
from .schemas import AgentRunResult, ClaimDecision
from .specialists import build_specialists
from .tracer import LangfuseTracer
from .tools import ToolRegistry

logger = logging.getLogger("shieldpoint_agents.manager")


# ---------------------------------------------------------------------------
# Keyword maps used by the classifier
# ---------------------------------------------------------------------------
_LIABILITY_KEYWORDS = ("injury", "attorney", "litigation", "bodily", "sued", "lawsuit")
_FRAUD_KEYWORDS = ("fraud", "intentional", "misrepresentation", "fake", "staged")
_THEFT_KEYWORDS = ("theft", "stolen", "burglary", "robbery", "larceny")
_AUTO_KEYWORDS = ("collision", "vehicle", "car ", "auto", "truck", "rear-end")
_PROPERTY_KEYWORDS = (
    "homeowners", "roof", "fence", "basement", "kitchen", "flood",
    "wind", "hail", "fire", "vandalism", "mailbox",
)


# ---------------------------------------------------------------------------
# ManagerAgent
# ---------------------------------------------------------------------------
class ManagerAgent(Agent):
    """Orchestrates specialist agents for multi-claim processing.

    Subclass of :class:`Agent` — inherits the ReAct loop, fallback
    engine, confidence scorer, and HITL escalator, but overrides
    :meth:`run` to delegate work to specialists rather than running its
    own ReAct loop. The base ReAct machinery is retained for cases
    where the manager needs to reason about an unexpected failure (e.g.
    re-routing after a specialist errors).

    Parameters
    ----------
    specialists : dict[str, Agent], optional
        Pre-built specialist agents keyed by name. If ``None``, the
        standard trio is built via :func:`build_specialists` using the
        provided ``config`` and ``llm_client_factory``.
    memory : EpisodicMemoryStore, optional
        Episodic memory store. Defaults to :class:`InMemoryEpisodicMemory`.
    conflict_resolver : ConflictResolver, optional
        Configurable conflict resolver. Defaults to the ``weighted``
        strategy with the standard priority map.
    max_parallel_workers : int
        Maximum parallel workers for parallel stages. Default 4.
    """

    def __init__(
        self,
        *,
        config: Optional[AgentConfig] = None,
        specialists: Optional[dict[str, Agent]] = None,
        memory: Optional[EpisodicMemoryStore] = None,
        conflict_resolver: Optional[ConflictResolver] = None,
        llm_client_factory: Optional[Callable[[], Any]] = None,
        fallback: Optional[FallbackEngine] = None,
        tracer: Optional[LangfuseTracer] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        hitl_escalator: Optional[HITLEscalator] = None,
        max_parallel_workers: int = 4,
    ) -> None:
        # The manager itself has an empty tool registry — it does not
        # run its own ReAct loop in normal operation. The base class
        # machinery is only used as a fallback.
        empty_registry = ToolRegistry()
        super().__init__(
            name="ManagerAgent",
            tools=empty_registry,
            tracer=tracer or LangfuseTracer(agent_name="ManagerAgent"),
            fallback=fallback or FallbackEngine(),
            config=config or AgentConfig.from_env(),
            llm_client=None,  # manager does not call the LLM directly
            system_prompt="(manager does not use the base ReAct prompt)",
            confidence_scorer=confidence_scorer,
            hitl_escalator=hitl_escalator,
        )

        self.specialists = specialists or build_specialists(
            config=self.config,
            llm_client_factory=llm_client_factory,
            fallback=self.fallback,
            confidence_scorer=self.confidence_scorer,
            hitl_escalator=self.hitl_escalator,
        )
        # Make sure every specialist shares the manager's tracer so all
        # spans attach to the same trace tree.
        for spec in self.specialists.values():
            spec.tracer = self.tracer
            if spec.tools.tracer is None:
                spec.tools.tracer = self.tracer

        self.memory = memory or InMemoryEpisodicMemory()
        self.conflict_resolver = conflict_resolver or ConflictResolver()
        self.max_parallel_workers = max_parallel_workers

    # ================================================================== #
    #  Public entry point — overridden from base Agent                   #
    # ================================================================== #
    def run(self, claim: dict[str, Any]) -> ManagerRunResult:  # type: ignore[override]
        """Orchestrate specialist agents to assess ``claim``.

        Wraps the whole orchestration in a single Langfuse trace so
        every specialist invocation, orchestration decision, and
        conflict resolution appears as a linked span in one tree.
        """
        claim_id = claim.get("claim_id", "<unknown>")
        logger.info(
            "ManagerAgent starting orchestration for claim %s", claim_id,
        )

        # Recall prior episodes for this claim (episodic memory).
        prior_episodes = self.memory.recall(claim_id) if claim_id != "<unknown>" else []
        memory_summary = self.memory.summarise_for_prompt(claim_id) if claim_id != "<unknown>" else ""
        memory_ids_used = [e.episode_id for e in prior_episodes]

        # Open the top-level trace.
        with self.tracer.trace(
            "manager_run",
            user_id=claim.get("adjuster_id"),
            session_id=claim.get("session_id"),
            metadata={
                "claim_id": claim_id,
                "manager.name": self.name,
                "memory.episodes_used": len(memory_ids_used),
                "memory.has_history": bool(memory_ids_used),
            },
            tags=["ManagerAgent", "orchestration"],
        ) as span:
            trace_id = getattr(span, "id", None) if span else None
            try:
                return self._orchestrate(
                    claim,
                    trace_id=trace_id,
                    memory_ids_used=memory_ids_used,
                    memory_summary=memory_summary,
                )
            except Exception as exc:
                # Catastrophic failure — fall back to the rule-based engine
                # so the manager always produces *some* answer.
                logger.exception(
                    "ManagerAgent orchestration failed for claim %s: %s",
                    claim_id, exc,
                )
                fb_result = self.fallback.run(
                    claim,
                    agent_name=self.name,
                    fallback_reason=f"manager_orchestration_failed: {exc!r}",
                )
                # Build a minimal plan so the result envelope is still valid.
                plan = OrchestrationPlan(
                    claim_type="unknown",
                    stages=[OrchestrationStage(
                        stage_id="fallback",
                        agent_names=["FallbackEngine"],
                        mode="sequential",
                        rationale="Manager orchestration failed; rule-based fallback engaged.",
                    )],
                    routing_rationale="fallback",
                    conflict_strategy=self.conflict_resolver.strategy,
                )
                return ManagerRunResult(
                    manager_name=self.name,
                    claim_id=claim_id,
                    decision=fb_result.decision,
                    source="fallback",
                    plan=plan,
                    invocations=[],
                    conflicts=[],
                    memory_entries_used=memory_ids_used,
                    trace_id=trace_id,
                    fallback_reason=fb_result.fallback_reason,
                    iterations=0,
                )

    # ================================================================== #
    #  Orchestration                                                      #
    # ================================================================== #
    def _orchestrate(
        self,
        claim: dict[str, Any],
        *,
        trace_id: Optional[str],
        memory_ids_used: list[str],
        memory_summary: str,
    ) -> ManagerRunResult:
        """Plan → invoke specialists → synthesise → record memory."""
        claim_id = claim.get("claim_id", "<unknown>")

        # ---- 1. Plan ----
        claim_type = self._classify_claim_type(claim)
        plan = self._build_plan_for_type(claim_type, claim)

        # Log the plan as a span (best-effort).
        self._trace_event(
            "orchestration_plan",
            metadata={
                "claim_type": claim_type,
                "stages": [s.model_dump() for s in plan.stages],
                "routing_rationale": plan.routing_rationale,
                "conflict_strategy": plan.conflict_strategy,
            },
        )

        # ---- 2. Execute stages ----
        invocations: list[AgentInvocationRecord] = []
        for stage_idx, stage in enumerate(plan.stages):
            self._trace_event(
                f"stage_start:{stage.stage_id}",
                metadata={
                    "stage_id": stage.stage_id,
                    "mode": stage.mode,
                    "agent_names": stage.agent_names,
                    "stage_index": stage_idx,
                },
            )
            stage_invocations = self._execute_stage(
                stage=stage,
                claim=claim,
                memory_summary=memory_summary,
                prior_invocations=invocations,
                trace_id=trace_id,
            )
            invocations.extend(stage_invocations)

            # Persist each specialist's output to episodic memory.
            for inv in stage_invocations:
                if inv.error is None and claim_id != "<unknown>":
                    entry = make_entry_from_result(
                        claim_id=claim_id,
                        agent_name=inv.agent_name,
                        result=inv.result,
                        trace_id=trace_id,
                        related_episode_ids=memory_ids_used,
                        metadata={
                            "stage_id": inv.stage_id,
                            "manager_run_trace_id": trace_id,
                        },
                    )
                    self.memory.append(entry)

        # ---- 3. Synthesise (with conflict resolution) ----
        resolution = self.conflict_resolver.resolve(
            invocations=invocations,
            strategy=plan.conflict_strategy,
            claim_id=claim_id,
        )

        # Log conflict resolution if any
        if resolution.record is not None:
            self._trace_event(
                "conflict_resolution",
                metadata=resolution.record.model_dump(),
            )

        # Decide source label
        if resolution.strategy == "escalation":
            source = "hitl_escalation"
        elif any(inv.error for inv in invocations) and not invocations:
            source = "fallback"
        else:
            source = "synthesised"

        # ---- 4. Build final result ----
        result = ManagerRunResult(
            manager_name=self.name,
            claim_id=claim_id,
            decision=resolution.decision,
            source=source,
            plan=plan,
            invocations=invocations,
            conflicts=[resolution.record] if resolution.record else [],
            memory_entries_used=memory_ids_used,
            trace_id=trace_id,
            fallback_reason=None,
            iterations=len(plan.stages),
        )
        logger.info(
            "ManagerAgent completed claim %s: decision=%s source=%s "
            "stages=%d invocations=%d conflicts=%d",
            claim_id, result.decision.decision, result.source,
            len(plan.stages), len(invocations), len(result.conflicts),
        )
        return result

    # ================================================================== #
    #  Stage execution                                                    #
    # ================================================================== #
    def _execute_stage(
        self,
        *,
        stage: OrchestrationStage,
        claim: dict[str, Any],
        memory_summary: str,
        prior_invocations: list[AgentInvocationRecord],
        trace_id: Optional[str],
    ) -> list[AgentInvocationRecord]:
        """Execute one orchestration stage.

        Sequential mode: run agents in listed order; each agent's claim
        dict is augmented with the prior agents' decisions (within the
        same stage AND across prior stages) and the episodic memory
        summary.

        Parallel mode: run all agents concurrently using a thread pool;
        no intra-stage data sharing (each only sees prior stages' summary).
        """
        if stage.mode == "parallel":
            return self._execute_stage_parallel(
                stage=stage, claim=claim, memory_summary=memory_summary,
                trace_id=trace_id,
            )
        return self._execute_stage_sequential(
            stage=stage, claim=claim, memory_summary=memory_summary,
            prior_invocations=prior_invocations, trace_id=trace_id,
        )

    def _execute_stage_sequential(
        self,
        *,
        stage: OrchestrationStage,
        claim: dict[str, Any],
        memory_summary: str,
        prior_invocations: list[AgentInvocationRecord],
        trace_id: Optional[str],
    ) -> list[AgentInvocationRecord]:
        results: list[AgentInvocationRecord] = []
        for agent_name in stage.agent_names:
            # Augment the claim with prior decisions (sequential data sharing)
            augmented_claim = self._augment_claim(
                claim, memory_summary=memory_summary,
                prior_invocations=prior_invocations + results,
            )
            inv = self._invoke_specialist(
                agent_name=agent_name,
                stage_id=stage.stage_id,
                claim=augmented_claim,
                trace_id=trace_id,
            )
            results.append(inv)
        return results

    def _execute_stage_parallel(
        self,
        *,
        stage: OrchestrationStage,
        claim: dict[str, Any],
        memory_summary: str,
        trace_id: Optional[str],
    ) -> list[AgentInvocationRecord]:
        # In parallel mode, every specialist sees the same augmented claim
        # (memory summary only — no intra-stage data sharing).
        augmented_claim = self._augment_claim(
            claim, memory_summary=memory_summary, prior_invocations=[],
        )
        results: list[AgentInvocationRecord] = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as pool:
            futures = {
                pool.submit(
                    self._invoke_specialist,
                    agent_name=name,
                    stage_id=stage.stage_id,
                    claim=augmented_claim,
                    trace_id=trace_id,
                ): name
                for name in stage.agent_names
            }
            for fut in as_completed(futures):
                results.append(fut.result())
        # Restore the order specified in the stage (for deterministic tests)
        ordered = sorted(
            results,
            key=lambda r: stage.agent_names.index(r.agent_name)
            if r.agent_name in stage.agent_names else 999,
        )
        return ordered

    # ================================================================== #
    #  Specialist invocation (one specialist, one stage)                  #
    # ================================================================== #
    def _invoke_specialist(
        self,
        *,
        agent_name: str,
        stage_id: str,
        claim: dict[str, Any],
        trace_id: Optional[str],
    ) -> AgentInvocationRecord:
        """Invoke a single specialist and wrap its result.

        The specialist's ``run`` opens its own nested Langfuse span (via
        its own ``tracer.trace("agent_run")`` call) which auto-attaches
        to the manager's open trace because we share the same tracer
        instance and OTel context vars propagate.
        """
        agent = self.specialists.get(agent_name)
        if agent is None:
            logger.error("Specialist %r not registered", agent_name)
            now = time.time()
            return AgentInvocationRecord(
                agent_name=agent_name,
                stage_id=stage_id,
                started_at=now,
                finished_at=now,
                result=self._error_result(claim, agent_name, f"specialist {agent_name!r} not registered"),
                error=f"specialist {agent_name!r} not registered",
                span_id=None,
            )

        started = time.time()
        try:
            self._trace_event(
                f"specialist_invoke:{agent_name}",
                metadata={
                    "agent_name": agent_name,
                    "stage_id": stage_id,
                    "claim_id": claim.get("claim_id"),
                },
            )
            result = agent.run(claim)
            finished = time.time()
            return AgentInvocationRecord(
                agent_name=agent_name,
                stage_id=stage_id,
                started_at=started,
                finished_at=finished,
                result=result,
                error=None,
                span_id=getattr(result, "trace_id", None),
            )
        except Exception as exc:
            finished = time.time()
            logger.exception(
                "Specialist %s raised during invocation for claim %s",
                agent_name, claim.get("claim_id"),
            )
            return AgentInvocationRecord(
                agent_name=agent_name,
                stage_id=stage_id,
                started_at=started,
                finished_at=finished,
                result=self._error_result(claim, agent_name, str(exc)),
                error=f"{type(exc).__name__}: {exc}",
                span_id=None,
            )

    # ================================================================== #
    #  Claim augmentation (episodic memory injection)                     #
    # ================================================================== #
    def _augment_claim(
        self,
        claim: dict[str, Any],
        *,
        memory_summary: str,
        prior_invocations: list[AgentInvocationRecord],
    ) -> dict[str, Any]:
        """Return a copy of ``claim`` with memory + prior decisions injected.

        Specialists receive:
        - ``claim["episodic_memory"]`` — string summary of prior episodes
        - ``claim["prior_specialist_decisions"]`` — list of
          ``{agent_name, decision, confidence, reasoning}`` dicts from
          earlier invocations (within the same manager run)
        """
        augmented = dict(claim)
        if memory_summary:
            augmented["episodic_memory"] = memory_summary
        if prior_invocations:
            augmented["prior_specialist_decisions"] = [
                {
                    "agent_name": inv.agent_name,
                    "decision": inv.result.decision.decision,
                    "confidence": inv.result.confidence_score,
                    "reasoning": inv.result.decision.reasoning,
                }
                for inv in prior_invocations
                if inv.error is None
            ]
        return augmented

    # ================================================================== #
    #  Routing — classify claim type + build plan                         #
    # ================================================================== #
    def _classify_claim_type(self, claim: dict[str, Any]) -> str:
        """Classify the claim into a high-level type for routing.

        Inspects the ``policy_id``, ``description``, and ``amount``
        fields. Returns one of: ``property``, ``liability``,
        ``auto_collision``, ``theft``, ``fraud_suspected``, ``unknown``.
        """
        description = str(claim.get("description", "")).lower()
        policy_id = str(claim.get("policy_id", "")).upper()

        # Fraud takes priority — even if other signals match.
        for kw in _FRAUD_KEYWORDS:
            if kw in description:
                return "fraud_suspected"

        # Liability (injury / attorney) — second highest priority
        for kw in _LIABILITY_KEYWORDS:
            if kw in description:
                return "liability"

        # Theft
        for kw in _THEFT_KEYWORDS:
            if kw in description:
                return "theft"

        # Auto collision (no injury)
        for kw in _AUTO_KEYWORDS:
            if kw in description or policy_id.startswith("AU"):
                return "auto_collision"

        # Property (homeowners)
        if policy_id.startswith("HO"):
            return "property"
        for kw in _PROPERTY_KEYWORDS:
            if kw in description:
                return "property"

        return "unknown"

    def _build_plan_for_type(
        self, claim_type: str, claim: dict[str, Any]
    ) -> OrchestrationPlan:
        """Build the :class:`OrchestrationPlan` for the given claim type.

        Routing rules are documented in the module docstring. Override
        in a subclass for product-line-specific tuning.
        """
        if claim_type == "property":
            stages = [OrchestrationStage(
                stage_id="stage-1",
                agent_names=["ClaimsAgent", "FinancialAgent", "SentimentAgent"],
                mode="parallel",
                rationale=(
                    "Property claims need concurrent coverage, financial, "
                    "and sentiment (fraud) checks — no inter-specialist "
                    "data dependency."
                ),
            )]
            routing_rationale = (
                "Homeowners / property claim — run all three specialists "
                "in parallel; sentiment checks for fraud signals."
            )
            strategy = "weighted"
        elif claim_type == "liability":
            stages = [
                OrchestrationStage(
                    stage_id="stage-1",
                    agent_names=["ClaimsAgent", "SentimentAgent"],
                    mode="parallel",
                    rationale=(
                        "Liability claims need policy validation + "
                        "sentiment/urgency assessment first."
                    ),
                ),
                OrchestrationStage(
                    stage_id="stage-2",
                    agent_names=["FinancialAgent"],
                    mode="sequential",
                    rationale=(
                        "Financial agent runs after liability findings "
                        "are available, so it can factor them in."
                    ),
                ),
            ]
            routing_rationale = (
                "Liability claim (injury/attorney) — run Claims + Sentiment "
                "first, then Financial with their outputs as context."
            )
            strategy = "priority"
        elif claim_type == "auto_collision":
            stages = [OrchestrationStage(
                stage_id="stage-1",
                agent_names=["ClaimsAgent", "FinancialAgent"],
                mode="parallel",
                rationale=(
                    "Routine auto collision — coverage + financial "
                    "checks only; skip sentiment (no fraud signal expected)."
                ),
            )]
            routing_rationale = (
                "Auto collision without injury — run Claims + Financial "
                "in parallel; sentiment not needed."
            )
            strategy = "weighted"
        elif claim_type == "theft":
            stages = [
                OrchestrationStage(
                    stage_id="stage-1",
                    agent_names=["SentimentAgent"],
                    mode="sequential",
                    rationale="Run sentiment first to flag potential fraud urgency.",
                ),
                OrchestrationStage(
                    stage_id="stage-2",
                    agent_names=["ClaimsAgent"],
                    mode="sequential",
                    rationale="Claims validation after sentiment context is available.",
                ),
                OrchestrationStage(
                    stage_id="stage-3",
                    agent_names=["FinancialAgent"],
                    mode="sequential",
                    rationale="Financial assessment last, with both prior specialists' context.",
                ),
            ]
            routing_rationale = (
                "Theft claim — sequential: Sentiment (fraud signals) → "
                "Claims (policy) → Financial (limits)."
            )
            strategy = "weighted"
        elif claim_type == "fraud_suspected":
            stages = [
                OrchestrationStage(
                    stage_id="stage-1",
                    agent_names=["SentimentAgent"],
                    mode="sequential",
                    rationale="Sentiment first to flag urgency/fraud markers.",
                ),
                OrchestrationStage(
                    stage_id="stage-2",
                    agent_names=["ClaimsAgent", "FinancialAgent"],
                    mode="parallel",
                    rationale=(
                        "Claims + Financial in parallel after sentiment "
                        "context is available."
                    ),
                ),
            ]
            routing_rationale = (
                "Fraud suspected — sequential Sentiment first, then "
                "Claims + Financial in parallel."
            )
            strategy = "priority"
        else:  # unknown
            stages = [OrchestrationStage(
                stage_id="stage-1",
                agent_names=["ClaimsAgent", "FinancialAgent", "SentimentAgent"],
                mode="parallel",
                rationale=(
                    "Unknown claim type — run all three in parallel as "
                    "a conservative default."
                ),
            )]
            routing_rationale = (
                "Unknown claim type — conservative default: all three "
                "specialists in parallel."
            )
            strategy = "weighted"

        return OrchestrationPlan(
            claim_type=claim_type,  # type: ignore[arg-type]
            stages=stages,
            routing_rationale=routing_rationale,
            conflict_strategy=strategy,  # type: ignore[arg-type]
        )

    # ================================================================== #
    #  Helpers                                                            #
    # ================================================================== #
    def _error_result(
        self, claim: dict[str, Any], agent_name: str, error: str,
    ) -> AgentRunResult:
        """Build a placeholder AgentRunResult for a failed specialist."""
        return AgentRunResult(
            agent_name=agent_name,
            claim_id=claim.get("claim_id"),
            decision=ClaimDecision(
                decision="route_to_manual_review",
                reasoning=f"Specialist {agent_name} errored: {error}",
                confidence=0.0,
                evidence=[],
            ),
            source="fallback",
            iterations=0,
            fallback_reason=error,
            trace_id=None,
        )

    def _trace_event(self, name: str, *, metadata: dict[str, Any]) -> None:
        """Best-effort: log a tracing event via the Langfuse client.

        We can't open a nested span without a context manager, but we
        can update the current trace's metadata via the underlying
        client (if available). Failures here are silently ignored —
        tracing must never break orchestration.
        """
        try:
            delegate = getattr(self.tracer, "_delegate", None)
            if delegate is None or not getattr(delegate, "enabled", False):
                return
            client = getattr(delegate, "client", None)
            if client is None:
                return
            # Open a quick span via the SDK and attach metadata.
            try:
                with client.start_as_current_span(
                    name=name, input=None, metadata=metadata,
                ) as sp:
                    if sp is not None:
                        sp.update(level="DEBUG")
            except Exception:
                # Fall back to trace metadata update if span creation fails.
                try:
                    client.update_current_trace(metadata=metadata)
                except Exception:
                    pass
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("trace_event %s failed: %s", name, exc)
