"""
Confidence scoring for the Tool-Using Agent (SHLD-14).

The confidence scorer evaluates LLM output along multiple dimensions:

1. **Self-assessed confidence** — the LLM's own confidence field in the
   ClaimDecision (0–1 scale). This is the primary signal.
2. **Output consistency** — whether the LLM's decision is stable across
   multiple ReAct iterations (if the agent flips between approve/deny,
   confidence drops).
3. **Evidence grounding** — whether the LLM cites specific evidence from
   tool outputs rather than making unsupported claims.
4. **Tool coverage** — whether the agent consulted the relevant tools
   before reaching a decision (e.g., validated the policy before approving).

The final confidence score is a weighted combination of these signals,
clamped to [0, 1]. When the score falls below the HITL threshold (0.85),
the claim is escalated to human review. When it falls below the fallback
threshold (0.50), the FallbackEngine takes over entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .schemas import ClaimDecision, ReActStep

logger = logging.getLogger("shieldpoint_agents.confidence")


@dataclass(frozen=True)
class ConfidenceWeights:
    """Weights for the confidence scoring dimensions.

    Must sum to 1.0. Adjust to tune the scorer's sensitivity.
    """

    self_assessed: float = 0.50
    consistency: float = 0.25
    evidence_grounding: float = 0.15
    tool_coverage: float = 0.10


@dataclass
class ConfidenceReport:
    """Detailed breakdown of a confidence score for audit/logging.

    Produced by :meth:`ConfidenceScorer.score` and attached to the
    agent's trace as metadata so auditors can see *why* a claim was
    escalated or auto-approved.
    """

    final_score: float
    self_assessed: float
    consistency: float
    evidence_grounding: float
    tool_coverage: float
    decision_history: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def is_below_hitl(self) -> bool:
        """True if the score is below the standard HITL threshold (0.85)."""
        return self.final_score < 0.85

    @property
    def is_below_fallback(self) -> bool:
        """True if the score is below the fallback threshold (0.50)."""
        return self.final_score < 0.50


class ConfidenceScorer:
    """Multi-signal confidence scorer for the Tool-Using Agent.

    Parameters
    ----------
    weights : ConfidenceWeights, optional
        Per-dimension weights. Defaults to the standard weights.
    required_tools : list[str], optional
        Tool names that the agent *should* invoke before reaching a
        decision. If the agent never calls these, the tool_coverage
        score drops. Defaults to ``["validate_policy"]``.
    """

    def __init__(
        self,
        *,
        weights: ConfidenceWeights | None = None,
        required_tools: list[str] | None = None,
    ) -> None:
        self.weights = weights if weights is not None else ConfidenceWeights()
        self.required_tools = required_tools if required_tools is not None else ["validate_policy"]

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def score(
        self,
        decision: ClaimDecision,
        *,
        step_history: list[ReActStep],
        tools_invoked: list[str],
        claim: dict[str, Any],
    ) -> ConfidenceReport:
        """Compute a multi-signal confidence score for a ClaimDecision.

        Parameters
        ----------
        decision : ClaimDecision
            The final decision from the LLM.
        step_history : list[ReActStep]
            All ReAct steps taken by the agent (including the final one).
        tools_invoked : list[str]
            Names of tools actually invoked during the run.
        claim : dict[str, Any]
            The original claim dict (used for context checks).

        Returns
        -------
        ConfidenceReport
            Detailed confidence breakdown.
        """
        self_assessed = self._score_self_assessed(decision)
        consistency = self._score_consistency(step_history)
        evidence = self._score_evidence_grounding(decision, step_history)
        tool_cov = self._score_tool_coverage(tools_invoked)

        w = self.weights
        raw_score = (
            w.self_assessed * self_assessed
            + w.consistency * consistency
            + w.evidence_grounding * evidence
            + w.tool_coverage * tool_cov
        )
        final_score = max(0.0, min(1.0, raw_score))

        # Collect flags for audit
        flags: list[str] = []
        if consistency < 0.5:
            flags.append("inconsistent_decisions_across_iterations")
        if evidence < 0.5:
            flags.append("weak_evidence_grounding")
        if tool_cov < 0.5:
            flags.append("missing_required_tools")
        if self_assessed < 0.5:
            flags.append("llm_self_assessed_low_confidence")

        # Build decision history for audit
        decision_history: list[str] = []
        for s in step_history:
            if s.is_final:
                act_input = s.action_input
                d = act_input.get("decision", "unknown") if isinstance(act_input, dict) else "unknown"
                decision_history.append(f"FINAL_ANSWER:{d}")
            else:
                decision_history.append(f"tool_call:{s.action}")

        report = ConfidenceReport(
            final_score=final_score,
            self_assessed=self_assessed,
            consistency=consistency,
            evidence_grounding=evidence,
            tool_coverage=tool_cov,
            decision_history=decision_history,
            flags=flags,
        )
        logger.info(
            "Confidence score: %.3f (self=%.3f, consistency=%.3f, "
            "evidence=%.3f, tool_cov=%.3f) flags=%s",
            final_score, self_assessed, consistency, evidence, tool_cov, flags,
        )
        return report

    # ------------------------------------------------------------------ #
    #  Scoring dimensions                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _score_self_assessed(decision: ClaimDecision) -> float:
        """Return the LLM's own confidence, clamped to [0, 1].

        This is the primary signal — if the LLM is uncertain, we should
        respect that even if other signals look OK.
        """
        return max(0.0, min(1.0, decision.confidence))

    @staticmethod
    def _score_consistency(step_history: list[ReActStep]) -> float:
        """Score based on whether the agent's decision was stable.

        If the agent reached a FINAL_ANSWER on the first or second
        iteration, that's a consistency signal (the LLM was decisive).
        If the agent oscillated between different decisions across
        multiple iterations, confidence drops.

        We look at all FINAL_ANSWER attempts in the history (there
        should be at most one since the loop terminates on the first
        FINAL_ANSWER, but parse retries might produce multiple).
        """
        if not step_history:
            return 0.0

        # Count non-final steps — more steps before a decision means
        # less consistency (the agent was uncertain for longer).
        non_final_count = sum(1 for s in step_history if not s.is_final)

        # Discount for repeated tool calls to the same tool — that
        # suggests the agent didn't learn from the first observation.
        tool_calls = [s.action for s in step_history if not s.is_final]
        if tool_calls:
            unique_ratio = len(set(tool_calls)) / len(tool_calls)
        else:
            unique_ratio = 1.0

        # Iterations-to-decision factor: 1-2 iters = 1.0, 3-4 = 0.8,
        # 5-6 = 0.6, 7+ = 0.4
        if non_final_count <= 2:
            iter_factor = 1.0
        elif non_final_count <= 4:
            iter_factor = 0.8
        elif non_final_count <= 6:
            iter_factor = 0.6
        else:
            iter_factor = 0.4

        # Combine: repeated same-tool calls hurt consistency
        score = iter_factor * (0.5 + 0.5 * unique_ratio)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_evidence_grounding(
        decision: ClaimDecision, step_history: list[ReActStep]
    ) -> float:
        """Score based on whether the decision cites evidence from tools.

        A well-grounded decision:
        - Has at least 1 evidence item
        - Evidence items reference tool outputs (e.g., "peril=wind is
          in policy.perils_covered")
        - The agent actually invoked tools (not just guessed)
        """
        evidence_count = len(decision.evidence)

        # No evidence at all — very weak grounding
        if evidence_count == 0:
            # But if the agent did invoke tools, give partial credit
            tool_steps = [s for s in step_history if not s.is_final]
            return 0.3 if tool_steps else 0.1

        # Evidence referencing tool output patterns
        tool_reference_patterns = [
            "perils_covered", "limit", "deductible", "prior_count",
            "claim_history", "policy", "coverage", "peril=", "amount",
            "zkp", "proof", "payment",
        ]
        grounded_count = sum(
            1 for e in decision.evidence
            if any(p in e.lower() for p in tool_reference_patterns)
        )

        # Ratio of grounded evidence
        grounding_ratio = grounded_count / evidence_count if evidence_count else 0

        # Bonus for more evidence items (up to 3)
        volume_bonus = min(evidence_count / 3.0, 1.0)

        score = 0.5 * grounding_ratio + 0.3 * volume_bonus + 0.2 * (1.0 if evidence_count >= 1 else 0.0)
        return max(0.0, min(1.0, score))

    def _score_tool_coverage(self, tools_invoked: list[str]) -> float:
        """Score based on whether required tools were consulted.

        If the agent approved/denied a claim without checking the policy,
        the tool_coverage score drops significantly. This prevents the
        LLM from "guessing" without doing due diligence.
        """
        if not self.required_tools:
            # No required tools configured — full coverage by default.
            return 1.0

        invoked_set = set(tools_invoked)
        required_set = set(self.required_tools)

        covered = required_set & invoked_set
        coverage_ratio = len(covered) / len(required_set)

        # Even partial coverage gives some credit
        return max(0.0, min(1.0, coverage_ratio))
