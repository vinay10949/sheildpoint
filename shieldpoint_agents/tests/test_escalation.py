"""Unit tests for HITLEscalator — HITL escalation and fallback triggers (SHLD-14)."""

from __future__ import annotations

import pytest

from shieldpoint_agents import ClaimDecision
from shieldpoint_agents.confidence import ConfidenceReport
from shieldpoint_agents.escalation import EscalationRecord, HITLEscalator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_report(score: float = 0.9, flags: list[str] | None = None) -> ConfidenceReport:
    return ConfidenceReport(
        final_score=score,
        self_assessed=score,
        consistency=score,
        evidence_grounding=score,
        tool_coverage=score,
        flags=flags or [],
    )


def _make_decision(decision: str = "approve", confidence: float = 0.9) -> ClaimDecision:
    return ClaimDecision(
        decision=decision,
        reasoning="Test reasoning.",
        confidence=confidence,
        evidence=["peril=wind is covered"],
    )


# ---------------------------------------------------------------------------
# Accept: confidence >= HITL threshold
# ---------------------------------------------------------------------------
class TestAcceptDecision:
    def test_high_confidence_accepted(self):
        escalator = HITLEscalator(hitl_threshold=0.85, fallback_threshold=0.50)
        decision = _make_decision(confidence=0.95)
        report = _make_report(score=0.95)
        adjusted, record = escalator.evaluate(decision, report, claim_id="CLM-1")
        assert record is None
        assert adjusted.decision == "approve"
        assert adjusted.confidence == 0.95

    def test_exactly_at_threshold_accepted(self):
        escalator = HITLEscalator(hitl_threshold=0.85, fallback_threshold=0.50)
        decision = _make_decision(confidence=0.85)
        report = _make_report(score=0.85)
        adjusted, record = escalator.evaluate(decision, report, claim_id="CLM-2")
        assert record is None
        assert adjusted.decision == "approve"


# ---------------------------------------------------------------------------
# Escalate: fallback_threshold <= confidence < HITL threshold
# ---------------------------------------------------------------------------
class TestHITLEscalation:
    def test_moderate_confidence_escalated(self):
        escalator = HITLEscalator(hitl_threshold=0.85, fallback_threshold=0.50)
        decision = _make_decision(confidence=0.7)
        report = _make_report(score=0.7)
        adjusted, record = escalator.evaluate(decision, report, claim_id="CLM-3")

        assert record is not None
        assert adjusted.decision == "route_to_manual_review"
        assert "HITL ESCALATION" in adjusted.reasoning
        assert adjusted.confidence == 0.7
        assert record.original_decision == "approve"
        assert record.adjusted_confidence == 0.7

    def test_escalation_preserves_evidence(self):
        escalator = HITLEscalator(hitl_threshold=0.85, fallback_threshold=0.50)
        decision = _make_decision(confidence=0.75)
        report = _make_report(score=0.75)
        adjusted, record = escalator.evaluate(decision, report, claim_id="CLM-4")

        assert adjusted.evidence[0] == "peril=wind is covered"
        assert any("escalation_trigger" in e for e in adjusted.evidence)

    def test_deny_escalated_to_manual_review(self):
        escalator = HITLEscalator(hitl_threshold=0.85, fallback_threshold=0.50)
        decision = _make_decision(decision="deny", confidence=0.6)
        report = _make_report(score=0.6)
        adjusted, record = escalator.evaluate(decision, report, claim_id="CLM-5")

        assert adjusted.decision == "route_to_manual_review"
        assert record.original_decision == "deny"

    def test_escalation_record_metadata(self):
        escalator = HITLEscalator(hitl_threshold=0.85, fallback_threshold=0.50)
        decision = _make_decision(confidence=0.7)
        report = _make_report(score=0.7, flags=["weak_evidence_grounding"])
        adjusted, record = escalator.evaluate(decision, report, claim_id="CLM-6")

        metadata = record.to_metadata()
        assert metadata["hitl_escalation.original_decision"] == "approve"
        assert metadata["hitl_escalation.adjusted_confidence"] == 0.7
        assert "weak_evidence_grounding" in metadata["hitl_escalation.flags"]


# ---------------------------------------------------------------------------
# Fallback: confidence < fallback_threshold
# ---------------------------------------------------------------------------
class TestFallbackTrigger:
    def test_very_low_confidence_triggers_fallback(self):
        escalator = HITLEscalator(hitl_threshold=0.85, fallback_threshold=0.50)
        decision = _make_decision(confidence=0.3)
        report = _make_report(score=0.3)
        adjusted, record = escalator.evaluate(decision, report, claim_id="CLM-7")

        assert record is not None
        assert escalator.should_fallback(record) is True

    def test_exactly_at_fallback_threshold_not_fallback(self):
        escalator = HITLEscalator(hitl_threshold=0.85, fallback_threshold=0.50)
        decision = _make_decision(confidence=0.5)
        report = _make_report(score=0.5)
        adjusted, record = escalator.evaluate(decision, report, claim_id="CLM-8")

        # At exactly 0.5, it should escalate (not fallback), since < 0.85
        # and >= 0.50
        assert record is not None
        assert escalator.should_fallback(record) is False
        assert adjusted.decision == "route_to_manual_review"

    def test_no_record_means_no_fallback(self):
        escalator = HITLEscalator(hitl_threshold=0.85, fallback_threshold=0.50)
        assert escalator.should_fallback(None) is False

    def test_zero_confidence_triggers_fallback(self):
        escalator = HITLEscalator(hitl_threshold=0.85, fallback_threshold=0.50)
        decision = _make_decision(confidence=0.0)
        report = _make_report(score=0.0)
        adjusted, record = escalator.evaluate(decision, report, claim_id="CLM-9")

        assert escalator.should_fallback(record) is True


# ---------------------------------------------------------------------------
# Custom thresholds
# ---------------------------------------------------------------------------
class TestCustomThresholds:
    def test_strict_threshold_catches_more(self):
        # Very strict: require 0.95 confidence
        escalator = HITLEscalator(hitl_threshold=0.95, fallback_threshold=0.70)
        decision = _make_decision(confidence=0.9)
        report = _make_report(score=0.9)
        adjusted, record = escalator.evaluate(decision, report, claim_id="CLM-10")

        # 0.9 < 0.95 → escalated
        assert record is not None
        assert adjusted.decision == "route_to_manual_review"

    def test_lenient_threshold_accepts_more(self):
        # Very lenient: only require 0.5 confidence
        escalator = HITLEscalator(hitl_threshold=0.50, fallback_threshold=0.20)
        decision = _make_decision(confidence=0.6)
        report = _make_report(score=0.6)
        adjusted, record = escalator.evaluate(decision, report, claim_id="CLM-11")

        # 0.6 >= 0.5 → accepted
        assert record is None
        assert adjusted.decision == "approve"
