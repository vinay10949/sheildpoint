"""Unit tests for the ConfidenceScorer — multi-signal confidence scoring (SHLD-14)."""

from __future__ import annotations

import json

import pytest

from shieldpoint_agents import ClaimDecision, ReActStep
from shieldpoint_agents.confidence import (
    ConfidenceReport,
    ConfidenceScorer,
    ConfidenceWeights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_step(action: str = "validate_policy", thought: str = "Checking policy.",
               action_input: dict | None = None) -> ReActStep:
    return ReActStep(
        thought=thought,
        action=action,
        action_input=action_input or {"policy_id": "HO-001"},
    )


def _make_final_step(decision: str = "approve", confidence: float = 0.9,
                      evidence: list[str] | None = None) -> ReActStep:
    return ReActStep(
        thought="Making final decision.",
        action="FINAL_ANSWER",
        action_input={
            "decision": decision,
            "reasoning": "Test reasoning.",
            "confidence": confidence,
            "evidence": evidence or ["peril=wind is in policy.perils_covered"],
        },
    )


def _make_decision(decision: str = "approve", confidence: float = 0.9,
                    evidence: list[str] | None = None) -> ClaimDecision:
    if evidence is None:
        evidence = ["peril=wind is in policy.perils_covered"]
    return ClaimDecision(
        decision=decision,
        reasoning="Test reasoning.",
        confidence=confidence,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Self-assessed confidence
# ---------------------------------------------------------------------------
class TestSelfAssessedScore:
    def test_high_self_assessed_gives_high_score(self):
        scorer = ConfidenceScorer()
        decision = _make_decision(confidence=0.95)
        report = scorer.score(decision, step_history=[], tools_invoked=[], claim={})
        assert report.self_assessed == 0.95

    def test_low_self_assessed_gives_low_score(self):
        scorer = ConfidenceScorer()
        decision = _make_decision(confidence=0.3)
        report = scorer.score(decision, step_history=[], tools_invoked=[], claim={})
        assert report.self_assessed == 0.3

    def test_self_assessed_clamped_to_1(self):
        scorer = ConfidenceScorer()
        # Pydantic enforces confidence <= 1.0, so test with exactly 1.0
        decision = _make_decision(confidence=1.0)
        report = scorer.score(decision, step_history=[], tools_invoked=[], claim={})
        assert report.self_assessed == 1.0


# ---------------------------------------------------------------------------
# Consistency scoring
# ---------------------------------------------------------------------------
class TestConsistencyScore:
    def test_quick_decision_gives_high_consistency(self):
        scorer = ConfidenceScorer()
        step = _make_step()
        final = _make_final_step()
        decision = _make_decision()
        report = scorer.score(decision, step_history=[step, final],
                              tools_invoked=["validate_policy"], claim={})
        # 1 non-final step → iter_factor=1.0
        assert report.consistency >= 0.8

    def test_many_iterations_gives_lower_consistency(self):
        scorer = ConfidenceScorer()
        steps = [_make_step() for _ in range(8)] + [_make_final_step()]
        decision = _make_decision()
        report = scorer.score(decision, step_history=steps,
                              tools_invoked=["validate_policy"] * 8, claim={})
        # 8 non-final steps → iter_factor=0.4
        assert report.consistency < 0.7

    def test_repeated_same_tool_hurts_consistency(self):
        scorer = ConfidenceScorer()
        # Same tool called 5 times → unique_ratio = 1/5 = 0.2
        steps = [_make_step(action="validate_policy") for _ in range(5)]
        steps.append(_make_final_step())
        decision = _make_decision()
        report = scorer.score(decision, step_history=steps,
                              tools_invoked=["validate_policy"] * 5, claim={})
        assert report.consistency < 0.8

    def test_empty_history_gives_zero_consistency(self):
        scorer = ConfidenceScorer()
        decision = _make_decision()
        report = scorer.score(decision, step_history=[], tools_invoked=[], claim={})
        assert report.consistency == 0.0


# ---------------------------------------------------------------------------
# Evidence grounding scoring
# ---------------------------------------------------------------------------
class TestEvidenceGroundingScore:
    def test_tool_referenced_evidence_scores_high(self):
        scorer = ConfidenceScorer()
        decision = _make_decision(evidence=[
            "peril=wind is in policy.perils_covered",
            "amount 1250.00 <= limit 25000.00",
            "prior_count=0",
        ])
        final = _make_final_step()
        report = scorer.score(decision, step_history=[final],
                              tools_invoked=["validate_policy"], claim={})
        assert report.evidence_grounding >= 0.7

    def test_no_evidence_scores_low(self):
        scorer = ConfidenceScorer()
        decision = _make_decision(evidence=[])
        step = _make_step()  # tool step so tool_steps is not empty
        final = _make_final_step()
        report = scorer.score(decision, step_history=[step, final],
                              tools_invoked=["validate_policy"], claim={})
        # With tools invoked but no evidence, partial credit (0.3)
        assert report.evidence_grounding == 0.3

    def test_no_evidence_no_tools_scores_very_low(self):
        scorer = ConfidenceScorer()
        decision = _make_decision(evidence=[])
        final = _make_final_step()
        report = scorer.score(decision, step_history=[final],
                              tools_invoked=[], claim={})
        # No tools in history, no evidence → very low (0.1)
        assert report.evidence_grounding == 0.1

    def test_generic_evidence_scores_medium(self):
        scorer = ConfidenceScorer()
        decision = _make_decision(evidence=[
            "The claim seems valid",
            "No red flags detected",
        ])
        final = _make_final_step()
        report = scorer.score(decision, step_history=[final],
                              tools_invoked=[], claim={})
        # Generic evidence without tool references
        assert 0.2 <= report.evidence_grounding <= 0.6


# ---------------------------------------------------------------------------
# Tool coverage scoring
# ---------------------------------------------------------------------------
class TestToolCoverageScore:
    def test_all_required_tools_invoked(self):
        scorer = ConfidenceScorer(required_tools=["validate_policy"])
        decision = _make_decision()
        final = _make_final_step()
        report = scorer.score(decision, step_history=[final],
                              tools_invoked=["validate_policy"], claim={})
        assert report.tool_coverage == 1.0

    def test_missing_required_tool(self):
        scorer = ConfidenceScorer(required_tools=["validate_policy", "check_claim_history"])
        decision = _make_decision()
        final = _make_final_step()
        report = scorer.score(decision, step_history=[final],
                              tools_invoked=["validate_policy"], claim={})
        # Only 1 of 2 required tools → 0.5
        assert report.tool_coverage == 0.5

    def test_no_required_tools_configured(self):
        scorer = ConfidenceScorer(required_tools=[])
        decision = _make_decision()
        final = _make_final_step()
        report = scorer.score(decision, step_history=[final],
                              tools_invoked=[], claim={})
        assert report.tool_coverage == 1.0


# ---------------------------------------------------------------------------
# Final score composition
# ---------------------------------------------------------------------------
class TestFinalScoreComposition:
    def test_high_confidence_across_all_dimensions(self):
        scorer = ConfidenceScorer()
        decision = _make_decision(confidence=0.95, evidence=[
            "peril=wind is in policy.perils_covered",
            "amount 1250.00 <= limit 25000.00",
        ])
        step = _make_step()
        final = _make_final_step(confidence=0.95, evidence=[
            "peril=wind is in policy.perils_covered",
            "amount 1250.00 <= limit 25000.00",
        ])
        report = scorer.score(
            decision,
            step_history=[step, final],
            tools_invoked=["validate_policy"],
            claim={"amount": 1250},
        )
        assert report.final_score >= 0.80
        assert not report.is_below_hitl

    def test_low_confidence_triggers_hitl_flag(self):
        scorer = ConfidenceScorer()
        decision = _make_decision(confidence=0.6, evidence=[])
        final = _make_final_step(confidence=0.6, evidence=[])
        report = scorer.score(
            decision,
            step_history=[final],
            tools_invoked=[],
            claim={},
        )
        assert report.is_below_hitl

    def test_very_low_confidence_triggers_fallback_flag(self):
        scorer = ConfidenceScorer()
        decision = _make_decision(confidence=0.2, evidence=[])
        final = _make_final_step(confidence=0.2, evidence=[])
        report = scorer.score(
            decision,
            step_history=[final],
            tools_invoked=[],
            claim={},
        )
        assert report.is_below_fallback

    def test_custom_weights_change_composition(self):
        # Make self_assessed weight = 1.0, all others = 0
        weights = ConfidenceWeights(
            self_assessed=1.0, consistency=0.0,
            evidence_grounding=0.0, tool_coverage=0.0,
        )
        scorer = ConfidenceScorer(weights=weights)
        decision = _make_decision(confidence=0.75, evidence=[])
        report = scorer.score(decision, step_history=[], tools_invoked=[], claim={})
        # Final score should equal self_assessed (0.75)
        assert abs(report.final_score - 0.75) < 0.01

    def test_score_is_clamped_between_0_and_1(self):
        scorer = ConfidenceScorer()
        decision = _make_decision(confidence=0.5, evidence=[])
        report = scorer.score(decision, step_history=[], tools_invoked=[], claim={})
        assert 0.0 <= report.final_score <= 1.0


# ---------------------------------------------------------------------------
# Flags and audit trail
# ---------------------------------------------------------------------------
class TestConfidenceFlags:
    def test_inconsistent_decisions_flagged(self):
        scorer = ConfidenceScorer()
        # 8+ iterations with repeated same tool
        steps = [_make_step(action="validate_policy") for _ in range(8)]
        final = _make_final_step(confidence=0.5)
        steps.append(final)
        decision = _make_decision(confidence=0.5)
        report = scorer.score(decision, step_history=steps,
                              tools_invoked=["validate_policy"] * 8, claim={})
        assert "inconsistent_decisions_across_iterations" in report.flags

    def test_weak_evidence_flagged(self):
        scorer = ConfidenceScorer()
        decision = _make_decision(confidence=0.9, evidence=[])
        step = _make_step()
        final = _make_final_step(confidence=0.9, evidence=[])
        report = scorer.score(decision, step_history=[step, final],
                              tools_invoked=["validate_policy"], claim={})
        assert "weak_evidence_grounding" in report.flags

    def test_missing_required_tools_flagged(self):
        scorer = ConfidenceScorer(required_tools=["validate_policy"])
        decision = _make_decision(confidence=0.9)
        final = _make_final_step()
        report = scorer.score(decision, step_history=[final],
                              tools_invoked=[], claim={})
        assert "missing_required_tools" in report.flags

    def test_decision_history_recorded(self):
        scorer = ConfidenceScorer()
        step = _make_step()
        final = _make_final_step()
        decision = _make_decision()
        report = scorer.score(decision, step_history=[step, final],
                              tools_invoked=["validate_policy"], claim={})
        assert len(report.decision_history) == 2
        assert report.decision_history[0] == "tool_call:validate_policy"
        assert report.decision_history[1].startswith("FINAL_ANSWER:")
