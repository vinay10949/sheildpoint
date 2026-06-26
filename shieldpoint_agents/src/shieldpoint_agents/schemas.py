"""
Pydantic models for structured ReAct output.

The Agent's LLM is asked to produce JSON conforming to :class:`ReActStep` at
each reasoning iteration. The final answer is a :class:`ClaimDecision`. Both
models use Pydantic v2 for validation, which gives us:

- Strict type checking (strings stay strings, numbers stay numbers).
- Helpful error messages when the LLM produces malformed JSON.
- ``model_dump()`` for serializing back to plain dicts for tracing.

The ``Agent`` class catches ``ValidationError`` and retries the LLM call with
a corrective prompt; after ``parse_retries`` failures, the FallbackEngine
takes over.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReActStep(BaseModel):
    """A single Think/Plan/Act step in the ReAct loop.

    The LLM must produce this exact shape:

    .. code-block:: json

        {
          "thought": "I need to check the policy limits first.",
          "action": "validate_policy",
          "action_input": {"policy_id": "HO-2024-001"}
        }

    The special ``action == "FINAL_ANSWER"`` ends the loop. In that case
    ``action_input`` must contain the keys of :class:`ClaimDecision`.
    """

    model_config = ConfigDict(extra="forbid")

    thought: str = Field(
        ..., description="The agent's reasoning about what to do next."
    )
    action: str = Field(
        ...,
        description=(
            "Tool name to invoke, or 'FINAL_ANSWER' to end the loop and "
            "return the result."
        ),
    )
    action_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool, or the final decision dict.",
    )

    @field_validator("thought")
    @classmethod
    def _nonempty_thought(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("thought must be non-empty")
        return v.strip()

    @property
    def is_final(self) -> bool:
        return self.action.upper() == "FINAL_ANSWER"


class ClaimDecision(BaseModel):
    """The final structured output of a successful agent run.

    Returned either by the LLM (when the ReAct loop completes) or by the
    :class:`FallbackEngine` (when the LLM is unavailable).
    """

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "deny", "route_to_manual_review"] = Field(
        ..., description="The claim disposition."
    )
    reasoning: str = Field(
        ..., description="Human-readable explanation of the decision."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score in [0, 1]."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Cited facts from tool outputs that support the decision.",
    )

    @field_validator("reasoning")
    @classmethod
    def _nonempty_reasoning(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("reasoning must be non-empty")
        return v.strip()


class AgentRunResult(BaseModel):
    """Top-level envelope returned by :meth:`Agent.run`.

    Wraps the :class:`ClaimDecision` with provenance metadata: which agent
    produced it, whether it came from the LLM or the fallback engine, and
    how many ReAct iterations ran. Enhanced in SHLD-14 to carry confidence
    and HITL escalation details.
    """

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    claim_id: str | None = None
    decision: ClaimDecision
    source: Literal["llm", "fallback", "hitl_escalation"] = Field(
        ...,
        description=(
            "'llm' if the ReAct loop completed with sufficient confidence; "
            "'fallback' if the FallbackEngine was used; 'hitl_escalation' if "
            "the decision was escalated to human review due to low confidence."
        ),
    )
    iterations: int = Field(
        ..., ge=0, description="Number of ReAct iterations executed."
    )
    fallback_reason: str | None = Field(
        default=None,
        description=(
            "If source=='fallback', the reason (timeout, parse error, etc.)."
        ),
    )
    trace_id: str | None = Field(
        default=None,
        description="Langfuse trace ID, if tracing is enabled.",
    )
    # ---- SHLD-14: Confidence & escalation metadata ----
    confidence_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Final confidence score from the multi-signal scorer.",
    )
    hitl_escalated: bool = Field(
        default=False,
        description="True if the claim was escalated to human review.",
    )
    original_decision: str | None = Field(
        default=None,
        description=(
            "The LLM's original decision before HITL escalation. Only set "
            "when hitl_escalated=True."
        ),
    )
    tools_invoked: list[str] = Field(
        default_factory=list,
        description="Names of tools invoked during the ReAct loop.",
    )
