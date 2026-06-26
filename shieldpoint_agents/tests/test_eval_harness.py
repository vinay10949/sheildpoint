"""Tests for the evaluation harness — batch claims evaluation (SHLD-14)."""

from __future__ import annotations

import json

import pytest

from shieldpoint_agents import (
    Agent,
    AgentConfig,
    ConfidenceScorer,
    FallbackEngine,
    HITLEscalator,
    LangfuseTracer,
    ToolRegistry,
)
from shieldpoint_agents._testing import FakeLMClient
from shieldpoint_agents.eval_harness import (
    EvaluationHarness,
    EvaluationReport,
    EvaluationResult,
    generate_historical_claims,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_eval_agent(responses: list) -> Agent:
    """Build an agent with a FakeLMClient for evaluation testing."""
    client = FakeLMClient(responses)
    cfg = AgentConfig(
        lm_studio_base_url="http://localhost:1234/v1",
        lm_studio_api_key="lm-studio",
        model="qwen-test",
        llm_timeout_sec=10.0,
        max_react_iterations=10,
        parse_retries=1,
    )
    registry = ToolRegistry()

    @registry.register(
        name="validate_policy",
        description="Validate a policy by ID.",
        schema={
            "type": "object",
            "properties": {"policy_id": {"type": "string"}},
            "required": ["policy_id"],
        },
    )
    def validate_policy(policy_id: str) -> dict:
        return {"policy_id": policy_id, "limit": 25_000, "perils_covered": ["wind"]}

    return Agent(
        name="eval-agent",
        tools=registry,
        tracer=LangfuseTracer(agent_name="eval-agent"),
        fallback=FallbackEngine(),
        config=cfg,
        llm_client=client,
    )


def _final_answer_json(decision: str = "approve", confidence: float = 0.9) -> str:
    return json.dumps({
        "thought": "Making decision.",
        "action": "FINAL_ANSWER",
        "action_input": {
            "decision": decision,
            "reasoning": "Test reasoning.",
            "confidence": confidence,
            "evidence": ["peril=wind is covered"],
        },
    })


# ---------------------------------------------------------------------------
# Historical claims generation
# ---------------------------------------------------------------------------
class TestHistoricalClaimsGeneration:
    def test_generates_100_claims(self):
        claims = generate_historical_claims(count=100)
        assert len(claims) == 100

    def test_all_claims_have_expected_decision(self):
        claims = generate_historical_claims(count=100)
        for claim in claims:
            assert "expected_decision" in claim
            assert claim["expected_decision"] in (
                "approve", "deny", "route_to_manual_review"
            )

    def test_deterministic_with_same_seed(self):
        claims1 = generate_historical_claims(count=10, seed=42)
        claims2 = generate_historical_claims(count=10, seed=42)
        for c1, c2 in zip(claims1, claims2):
            assert c1["claim_id"] == c2["claim_id"]
            assert c1["description"] == c2["description"]

    def test_different_seeds_produce_different_claims(self):
        claims1 = generate_historical_claims(count=10, seed=42)
        claims2 = generate_historical_claims(count=10, seed=99)
        amounts1 = [c["amount"] for c in claims1]
        amounts2 = [c["amount"] for c in claims2]
        # At least some amounts should differ
        assert amounts1 != amounts2

    def test_claims_have_required_fields(self):
        claims = generate_historical_claims(count=10)
        for claim in claims:
            assert "claim_id" in claim
            assert "amount" in claim
            assert "description" in claim
            assert "policy_id" in claim
            assert "claimant" in claim


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------
class TestEvaluationResult:
    def test_to_dict(self):
        result = EvaluationResult(
            claim_id="CLM-1",
            predicted_decision="approve",
            expected_decision="approve",
            match=True,
            source="llm",
            confidence_score=0.9,
            iterations=2,
            hitl_escalated=False,
        )
        d = result.to_dict()
        assert d["claim_id"] == "CLM-1"
        assert d["match"] is True

    def test_mismatch_detected(self):
        result = EvaluationResult(
            claim_id="CLM-2",
            predicted_decision="approve",
            expected_decision="deny",
            match=False,
            source="llm",
            confidence_score=0.6,
            iterations=3,
            hitl_escalated=True,
        )
        assert result.match is False


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------
class TestEvaluationReport:
    def test_accuracy_calculation(self):
        report = EvaluationReport(total_claims=10, correct=8, incorrect=2, errors=0)
        assert report.accuracy == 0.8

    def test_accuracy_with_errors(self):
        report = EvaluationReport(total_claims=10, correct=7, incorrect=2, errors=1)
        # accuracy = correct / (total - errors) = 7/9
        assert abs(report.accuracy - 7 / 9) < 0.01

    def test_zero_claims_accuracy(self):
        report = EvaluationReport(total_claims=0)
        assert report.accuracy == 0.0

    def test_to_dict(self):
        report = EvaluationReport(
            total_claims=5,
            correct=4,
            incorrect=1,
            errors=0,
            avg_confidence=0.85,
            avg_iterations=2.5,
        )
        d = report.to_dict()
        assert d["total_claims"] == 5
        assert d["accuracy"] == 0.8

    def test_summary_string(self):
        report = EvaluationReport(
            total_claims=100,
            correct=92,
            incorrect=8,
            errors=0,
            avg_confidence=0.88,
        )
        summary = report.summary()
        assert "100 claims" in summary
        assert "92.0%" in summary


# ---------------------------------------------------------------------------
# EvaluationHarness — small batch test
# ---------------------------------------------------------------------------
class TestEvaluationHarnessSmallBatch:
    def test_small_batch_all_approved(self):
        """Simulate evaluating 5 claims where agent approves all."""
        claims = generate_historical_claims(count=5)
        # Generate responses: 1 FINAL_ANSWER per claim
        responses = [_final_answer_json(decision="approve", confidence=0.9)] * 5
        agent = _build_eval_agent(responses)
        harness = EvaluationHarness(agent, claims)
        report = harness.run()

        assert report.total_claims == 5
        assert report.avg_confidence > 0.8
        assert report.source_distribution.get("llm", 0) + \
               report.source_distribution.get("hitl_escalation", 0) + \
               report.source_distribution.get("fallback", 0) == 5

    def test_harness_handles_errors(self):
        """Agent fails on all claims — errors counted."""
        claims = generate_historical_claims(count=3)
        # Force errors by providing bad responses that trigger fallback
        agent = _build_eval_agent([TimeoutError("timeout")] * 3)
        harness = EvaluationHarness(agent, claims)
        report = harness.run()

        assert report.total_claims == 3
        # All should be fallback
        assert report.fallback_count == 3

    def test_mixed_results_tracked(self):
        """Some approve, some escalate — all tracked."""
        claims = [
            {"claim_id": "CLM-1", "amount": 100, "description": "Minor wind damage.",
             "expected_decision": "approve"},
            {"claim_id": "CLM-2", "amount": 5000, "description": "Injury claim.",
             "expected_decision": "route_to_manual_review"},
        ]
        # First: high confidence approve, Second: low confidence → escalation
        responses = [
            _final_answer_json(decision="approve", confidence=0.95),
            _final_answer_json(decision="approve", confidence=0.6),
        ]
        agent = _build_eval_agent(responses)
        harness = EvaluationHarness(agent, claims)
        report = harness.run()

        assert report.total_claims == 2
        # First should be correct (approve == approve), second may escalate
        # The second claim expects route_to_manual_review, agent says approve
        # but confidence is 0.6 so it may get escalated to route_to_manual_review
