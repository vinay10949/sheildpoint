"""
HITL (Human-In-The-Loop) escalation module (SHLD-14).

When the LLM's confidence score drops below the HITL threshold (0.85 by
default), the claim is *escalated* to a human reviewer rather than being
auto-approved or auto-denied. This module:

1. Detects when escalation is needed (confidence < hitl_confidence_threshold).
2. Transforms the ClaimDecision into a ``route_to_manual_review`` decision
   with full audit metadata explaining *why* escalation was triggered.
3. When confidence drops below the fallback threshold (0.50), the
   FallbackEngine takes over entirely instead of just escalating.

The escalation path preserves the LLM's original reasoning and confidence
so the human reviewer can see what the agent was thinking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .confidence import ConfidenceReport
from .schemas import AgentRunResult, ClaimDecision

logger = logging.getLogger("shieldpoint_agents.escalation")


@dataclass(frozen=True)
class EscalationRecord:
    """Audit record for a HITL escalation event.

    Attached to the Langfuse trace as metadata so reviewers can see
    exactly why a claim was escalated.
    """

    original_decision: str
    original_confidence: float
    adjusted_confidence: float
    reason: str
    confidence_flags: list[str]

    def to_metadata(self) -> dict[str, Any]:
        """Convert to a flat dict suitable for Langfuse trace metadata."""
        return {
            "hitl_escalation.original_decision": self.original_decision,
            "hitl_escalation.original_confidence": self.original_confidence,
            "hitl_escalation.adjusted_confidence": self.adjusted_confidence,
            "hitl_escalation.reason": self.reason,
            "hitl_escalation.flags": ", ".join(self.confidence_flags),
        }


class HITLEscalator:
    """Handles HITL escalation decisions for the Tool-Using Agent.

    The escalator is invoked after the ReAct loop produces a ClaimDecision.
    It checks the confidence report against thresholds and decides whether
    to:

    - **Accept** the LLM's decision (confidence >= 0.85)
    - **Escalate** to human review (0.50 <= confidence < 0.85)
    - **Fall back** to rule-based processing (confidence < 0.50)

    Parameters
    ----------
    hitl_threshold : float
        Minimum confidence to auto-approve/deny. Below this, escalate.
    fallback_threshold : float
        Minimum confidence to keep the LLM decision at all. Below this,
        the FallbackEngine takes over.
    """

    def __init__(
        self,
        *,
        hitl_threshold: float = 0.85,
        fallback_threshold: float = 0.50,
    ) -> None:
        self.hitl_threshold = hitl_threshold
        self.fallback_threshold = fallback_threshold

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        decision: ClaimDecision,
        confidence_report: ConfidenceReport,
        *,
        claim_id: str | None = None,
    ) -> tuple[ClaimDecision, EscalationRecord | None]:
        """Evaluate whether the LLM's decision should stand or be escalated.

        Returns
        -------
        tuple[ClaimDecision, EscalationRecord | None]
            The (possibly adjusted) decision and an optional escalation
            record (None if no escalation was needed).
        """
        score = confidence_report.final_score

        # Case 1: Confidence is high enough — accept the LLM's decision.
        if score >= self.hitl_threshold:
            logger.info(
                "Confidence %.3f >= HITL threshold %.3f — accepting LLM "
                "decision '%s' for claim %s",
                score, self.hitl_threshold, decision.decision, claim_id,
            )
            return decision, None

        # Case 2: Confidence is very low — trigger full fallback.
        if score < self.fallback_threshold:
            logger.warning(
                "Confidence %.3f < fallback threshold %.3f — triggering "
                "rule-based fallback for claim %s (original decision: '%s')",
                score, self.fallback_threshold, claim_id, decision.decision,
            )
            # The fallback engine will be invoked by the Agent — we signal
            # this by returning a special escalation record.
            record = EscalationRecord(
                original_decision=decision.decision,
                original_confidence=decision.confidence,
                adjusted_confidence=score,
                reason=(
                    f"Confidence score {score:.3f} is below fallback threshold "
                    f"{self.fallback_threshold:.3f}. Triggering rule-based "
                    "fallback per SHLD-14 AC."
                ),
                confidence_flags=confidence_report.flags,
            )
            # Return the original decision but flag it — the Agent will
            # invoke the FallbackEngine.
            return decision, record

        # Case 3: Confidence is between thresholds — escalate to HITL.
        # The decision becomes "route_to_manual_review" but we preserve
        # the original reasoning and evidence for the human reviewer.
        logger.info(
            "Confidence %.3f between fallback (%.3f) and HITL (%.3f) "
            "thresholds — escalating claim %s to manual review "
            "(original decision: '%s')",
            score, self.fallback_threshold, self.hitl_threshold,
            claim_id, decision.decision,
        )

        escalated = ClaimDecision(
            decision="route_to_manual_review",
            reasoning=(
                f"[HITL ESCALATION] Original LLM decision was "
                f"'{decision.decision}' at confidence {score:.3f} "
                f"(below HITL threshold {self.hitl_threshold:.3f}). "
                f"Original reasoning: {decision.reasoning}"
            ),
            confidence=score,
            evidence=decision.evidence + [
                f"escalation_trigger: confidence {score:.3f} < "
                f"HITL threshold {self.hitl_threshold:.3f}",
            ],
        )

        record = EscalationRecord(
            original_decision=decision.decision,
            original_confidence=decision.confidence,
            adjusted_confidence=score,
            reason=(
                f"Confidence score {score:.3f} is between fallback threshold "
                f"{self.fallback_threshold:.3f} and HITL threshold "
                f"{self.hitl_threshold:.3f}. Escalating to manual review."
            ),
            confidence_flags=confidence_report.flags,
        )

        return escalated, record

    def should_fallback(self, record: EscalationRecord | None) -> bool:
        """Return True if the escalation record indicates full fallback.

        Used by the Agent to decide whether to invoke the FallbackEngine.
        """
        if record is None:
            return False
        return record.adjusted_confidence < self.fallback_threshold
