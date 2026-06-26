"""Unit tests for FallbackEngine rule logic."""

from __future__ import annotations

import pytest

from shieldpoint_agents import FallbackEngine
from shieldpoint_agents.fallback import FallbackConfig


# ---------------------------------------------------------------------------
# Rule 1: explicit deny keywords
# ---------------------------------------------------------------------------
class TestDenyRule:
    def test_fraud_keyword_triggers_deny(self):
        engine = FallbackEngine()
        result = engine.evaluate({
            "claim_id": "CLM-1",
            "amount": 100.0,
            "description": "Suspected fraud — claimant has history of fake claims.",
        })
        assert result.decision.decision == "deny"
        assert result.decision.confidence == 0.95
        assert result.rule_name == "deny_keyword"

    def test_intentional_keyword_triggers_deny(self):
        engine = FallbackEngine()
        result = engine.evaluate({
            "claim_id": "CLM-2",
            "amount": 1000.0,
            "description": "Intentional damage to property.",
        })
        assert result.decision.decision == "deny"


# ---------------------------------------------------------------------------
# Rule 2: small-amount auto-approve
# ---------------------------------------------------------------------------
class TestAutoApproveRule:
    def test_small_amount_no_keywords_auto_approves(self):
        engine = FallbackEngine()
        result = engine.evaluate({
            "claim_id": "CLM-3",
            "amount": 250.00,
            "description": "Minor wind damage to mailbox.",
        })
        assert result.decision.decision == "approve"
        assert result.decision.confidence == 0.80
        assert result.rule_name == "small_amount_auto_approve"

    def test_small_amount_with_injury_keyword_routes_to_review(self):
        engine = FallbackEngine()
        result = engine.evaluate({
            "claim_id": "CLM-4",
            "amount": 250.00,
            "description": "Slip and injury on front steps.",
        })
        # injury keyword should override the auto-approve
        assert result.decision.decision == "route_to_manual_review"
        assert result.rule_name == "manual_review_trigger"

    def test_custom_threshold_respected(self):
        engine = FallbackEngine(FallbackConfig(auto_approve_threshold=100.0))
        result = engine.evaluate({
            "claim_id": "CLM-5",
            "amount": 250.00,
            "description": "Minor damage.",
        })
        # 250 > 100 → no auto-approve → default manual review
        assert result.decision.decision == "route_to_manual_review"


# ---------------------------------------------------------------------------
# Rule 3: high-amount or review keywords → manual review
# ---------------------------------------------------------------------------
class TestManualReviewRule:
    def test_high_amount_triggers_review(self):
        engine = FallbackEngine()
        result = engine.evaluate({
            "claim_id": "CLM-6",
            "amount": 7_500.00,
            "description": "Kitchen fire damage.",
        })
        assert result.decision.decision == "route_to_manual_review"
        assert result.rule_name == "manual_review_trigger"

    def test_litigation_keyword_triggers_review(self):
        engine = FallbackEngine()
        result = engine.evaluate({
            "claim_id": "CLM-7",
            "amount": 1_000.00,
            "description": "Claimant threatened litigation.",
        })
        assert result.decision.decision == "route_to_manual_review"
        assert "litigation" in result.reason


# ---------------------------------------------------------------------------
# Rule 4: default
# ---------------------------------------------------------------------------
class TestDefaultRule:
    def test_mid_amount_no_keywords_defaults_to_review(self):
        engine = FallbackEngine()
        result = engine.evaluate({
            "claim_id": "CLM-8",
            "amount": 1_500.00,
            "description": "Generic damage, no special markers.",
        })
        # 1500 is between auto_approve_threshold (500) and manual_review_threshold (5000)
        # and no keywords → default rule fires
        assert result.decision.decision == "route_to_manual_review"
        assert result.rule_name == "default_manual_review"
        assert result.decision.confidence == 0.40


# ---------------------------------------------------------------------------
# Full AgentRunResult envelope via .run()
# ---------------------------------------------------------------------------
class TestRunEnvelope:
    def test_run_returns_envelope_with_source_fallback(self):
        engine = FallbackEngine()
        result = engine.run(
            {"claim_id": "CLM-9", "amount": 100.0, "description": "small claim"},
            agent_name="test-agent",
            fallback_reason="llm_timeout",
        )
        assert result.source == "fallback"
        assert result.agent_name == "test-agent"
        assert result.claim_id == "CLM-9"
        assert "llm_timeout" in (result.fallback_reason or "")
        assert result.iterations == 0  # fallback doesn't run the loop
        assert result.decision.decision in ("approve", "deny", "route_to_manual_review")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_missing_amount_treated_as_inf(self):
        engine = FallbackEngine()
        result = engine.evaluate({"claim_id": "CLM-10", "description": "no amount"})
        # inf >= manual_review_threshold → manual review
        assert result.decision.decision == "route_to_manual_review"

    def test_missing_description_treated_as_empty(self):
        engine = FallbackEngine()
        result = engine.evaluate({"claim_id": "CLM-11", "amount": 100.0})
        # No keywords, small amount → auto-approve
        assert result.decision.decision == "approve"

    def test_case_insensitive_keyword_match(self):
        engine = FallbackEngine()
        result = engine.evaluate({
            "claim_id": "CLM-12",
            "amount": 100.0,
            "description": "FRAUD detected",
        })
        assert result.decision.decision == "deny"
