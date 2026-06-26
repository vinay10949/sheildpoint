"""
Pydantic models for multi-agent orchestration (SHLD-15 — ManagerAgent).

These schemas extend the base ``schemas.py`` models with the metadata needed
to coordinate three specialist agents (ClaimsAgent, FinancialAgent,
SentimentAgent) under a single ManagerAgent.

The orchestration flow is:

1. ManagerAgent receives a claim and produces an :class:`OrchestrationPlan`
   describing which specialists to invoke, in what sequence, and in what
   execution mode (sequential or parallel).
2. Each specialist's :class:`AgentRunResult` is captured as an
   :class:`AgentInvocationRecord` together with timing and tracing metadata.
3. If the specialist decisions disagree (per the :class:`ConflictDetector`
   rules), a :class:`ConflictRecord` is produced and fed to the
   :class:`ConflictResolver` which picks a resolution strategy.
4. The final synthesised :class:`ManagerRunResult` is returned, embedding
   the orchestration plan, all specialist results, any conflict records,
   and the final unified :class:`ClaimDecision`.

All models use Pydantic v2 with strict validation — same conventions as
``schemas.py``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schemas import AgentRunResult, ClaimDecision


# ---------------------------------------------------------------------------
# Orchestration plan
# ---------------------------------------------------------------------------
ExecutionMode = Literal["sequential", "parallel"]
"""How the specialists in a plan stage should be executed."""

ClaimType = Literal[
    "property",
    "liability",
    "auto_collision",
    "theft",
    "fraud_suspected",
    "unknown",
]
"""High-level claim category used to drive orchestration routing."""


class OrchestrationStage(BaseModel):
    """A single stage in an :class:`OrchestrationPlan`.

    A stage groups one or more specialist agent invocations that should be
    executed together (either sequentially or in parallel). The plan is a
    list of stages, run in order — later stages can see earlier stages'
    results via the episodic memory store.
    """

    model_config = ConfigDict(extra="forbid")

    stage_id: str = Field(
        ..., description="Stable identifier (e.g. 'stage-1')."
    )
    agent_names: list[str] = Field(
        ..., min_length=1,
        description="Specialist agent names to invoke in this stage.",
    )
    mode: ExecutionMode = Field(
        default="sequential",
        description="Execution mode for the agents in this stage.",
    )
    rationale: str = Field(
        default="",
        description="Why this stage exists (audit/commentary).",
    )


class OrchestrationPlan(BaseModel):
    """The orchestration plan produced by ``ManagerAgent._plan``.

    Captures the chain of stages the manager will execute, along with the
    claim-type classification that drove the routing decision.
    """

    model_config = ConfigDict(extra="forbid")

    claim_type: ClaimType = Field(
        ..., description="The claim category used for routing.",
    )
    stages: list[OrchestrationStage] = Field(
        ..., min_length=1,
        description="Ordered list of execution stages.",
    )
    routing_rationale: str = Field(
        default="",
        description="Why this plan was chosen for this claim type.",
    )
    conflict_strategy: Literal["priority", "vote", "escalation", "weighted"] = Field(
        default="weighted",
        description=(
            "Conflict-resolution strategy to apply when specialists disagree."
        ),
    )

    @property
    def all_agent_names(self) -> list[str]:
        """Flat list of every agent name across all stages (order-preserving)."""
        names: list[str] = []
        for stage in self.stages:
            for n in stage.agent_names:
                if n not in names:
                    names.append(n)
        return names


# ---------------------------------------------------------------------------
# Agent invocation record (per-specialist outcome)
# ---------------------------------------------------------------------------
class AgentInvocationRecord(BaseModel):
    """Record of one specialist agent's invocation inside a manager run.

    Wraps the specialist's :class:`AgentRunResult` with manager-level
    metadata: which stage it ran in, when, and the Langfuse span id that
    links it into the parent trace tree.
    """

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    stage_id: str
    started_at: float = Field(
        ..., description="Unix epoch seconds when the invocation started.",
    )
    finished_at: float = Field(
        ..., description="Unix epoch seconds when the invocation finished.",
    )
    result: AgentRunResult
    span_id: str | None = Field(
        default=None,
        description="Langfuse span id (if tracing is enabled).",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the specialist raised.",
    )

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    @property
    def decision_label(self) -> str:
        """The specialist's final decision label (or 'error' if it failed)."""
        if self.error is not None:
            return "error"
        return self.result.decision.decision


# ---------------------------------------------------------------------------
# Conflict records
# ---------------------------------------------------------------------------
class ConflictRecord(BaseModel):
    """Describes a disagreement between specialist agents.

    A conflict exists when two or more specialists return *different*
    decision labels (e.g. SentimentAgent says ``approve`` because the
    description is calm, but FinancialAgent says ``deny`` because the
    claim exceeds the coverage limit). The :class:`ConflictResolver`
    consumes one or more :class:`ConflictRecord` instances and produces
    a final synthesised :class:`ClaimDecision`.
    """

    model_config = ConfigDict(extra="forbid")

    conflict_id: str
    agent_names: list[str] = Field(
        ..., min_length=2,
        description="The agents whose decisions disagree.",
    )
    decisions: dict[str, str] = Field(
        ...,
        description=(
            "Map of agent_name → decision label. e.g. "
            "{'FinancialAgent':'deny','SentimentAgent:'approve'}."
        ),
    )
    description: str = Field(
        ..., description="Human-readable explanation of the conflict.",
    )
    strategy_used: Literal[
        "priority", "vote", "escalation", "weighted"
    ] = Field(
        ..., description="Which resolution strategy was applied.",
    )
    resolution: str = Field(
        ..., description="The resulting decision label chosen by the strategy.",
    )
    resolution_rationale: str = Field(
        ...,
        description=(
            "Auditable explanation of why this resolution was chosen."
        ),
    )


# ---------------------------------------------------------------------------
# Episodic memory entry
# ---------------------------------------------------------------------------
class EpisodicMemoryEntry(BaseModel):
    """One record in the episodic memory store.

    Stored as a row in the ``agent_episodes`` PostgreSQL table (JSONB
    payload) or its in-memory equivalent. Each entry captures a single
    specialist's contribution to a claim so that follow-up interactions
    on the same claim can reference prior agent outputs.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(
        ..., description="Unique episode id (UUID or similar).",
    )
    claim_id: str = Field(..., description="The claim this episode belongs to.")
    agent_name: str = Field(
        ..., description="Specialist or manager that produced this episode.",
    )
    decision_label: str
    decision: ClaimDecision
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    trace_id: str | None = None
    created_at: float = Field(
        ..., description="Unix epoch seconds.",
    )
    related_episode_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Final synthesised manager result
# ---------------------------------------------------------------------------
class ManagerRunResult(BaseModel):
    """Final envelope returned by :meth:`ManagerAgent.run`.

    Wraps the unified :class:`ClaimDecision` with the full orchestration
    provenance: the plan, every specialist's invocation record, any
    conflict records, the episodic memory snapshot used, and the Langfuse
    trace id that links everything together.
    """

    model_config = ConfigDict(extra="forbid")

    manager_name: str
    claim_id: str | None = None
    decision: ClaimDecision
    source: Literal["llm", "fallback", "hitl_escalation", "synthesised"] = Field(
        ...,
        description=(
            "'synthesised' when the manager combined specialist outputs; "
            "'fallback' when the manager fell back to rule-based; "
            "'hitl_escalation' when escalated via the conflict resolver."
        ),
    )
    plan: OrchestrationPlan
    invocations: list[AgentInvocationRecord] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    memory_entries_used: list[str] = Field(
        default_factory=list,
        description=(
            "Ids of prior episodic memory entries referenced for this run "
            "(empty on first interaction with a claim)."
        ),
    )
    trace_id: str | None = None
    fallback_reason: str | None = None
    iterations: int = Field(
        default=0, ge=0,
        description="Number of orchestration stages executed.",
    )

    @property
    def has_conflict(self) -> bool:
        return len(self.conflicts) > 0
