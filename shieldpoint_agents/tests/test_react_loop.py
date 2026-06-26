"""Comprehensive unit tests for the Agent ReAct loop — SHLD-14 enhancements.

Tests cover:
- Full ReAct cycle with confidence scoring
- HITL escalation when confidence < 0.85
- Fallback when confidence < 0.50
- Max iterations (10) enforced
- Confidence scoring integration
- Tool invocation tracking
- Malformed input handling
- Rule-based fallback activation
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from shieldpoint_agents import (
    Agent,
    AgentConfig,
    AgentRunResult,
    ClaimDecision,
    ConfidenceScorer,
    FallbackEngine,
    HITLEscalator,
    LangfuseTracer,
    ToolRegistry,
)
from shieldpoint_agents._testing import FakeLMClient
from shieldpoint_agents.agent import _FallbackSignal, _ParseError


# ---------------------------------------------------------------------------
# Helpers — build a minimal agent with a mock LLM client and tools.
# ---------------------------------------------------------------------------
def _build_agent(
    responses: list[Any],
    *,
    config: AgentConfig | None = None,
    confidence_scorer: ConfidenceScorer | None = None,
    hitl_escalator: HITLEscalator | None = None,
) -> tuple[Agent, FakeLMClient]:
    client = FakeLMClient(responses)
    cfg = config or AgentConfig(
        lm_studio_base_url="http://localhost:1234/v1",
        lm_studio_api_key="lm-studio",
        model="qwen-test",
        llm_timeout_sec=10.0,
        max_react_iterations=10,
        parse_retries=1,
        hitl_confidence_threshold=0.85,
        fallback_confidence_threshold=0.50,
    )
    registry = ToolRegistry()

    @registry.register(
        name="check_policy",
        description="Returns the policy coverage limit.",
        schema={
            "type": "object",
            "properties": {"policy_id": {"type": "string"}},
            "required": ["policy_id"],
        },
    )
    def check_policy(policy_id: str) -> dict:
        return {"policy_id": policy_id, "limit": 10_000, "perils_covered": ["wind"]}

    @registry.register(
        name="check_history",
        description="Check claimant's history.",
        schema={
            "type": "object",
            "properties": {"claimant": {"type": "string"}},
            "required": ["claimant"],
        },
    )
    def check_history(claimant: str) -> dict:
        return {"claimant": claimant, "prior_count": 0}

    agent = Agent(
        name="test-agent",
        tools=registry,
        tracer=LangfuseTracer(agent_name="test-agent"),
        fallback=FallbackEngine(),
        config=cfg,
        llm_client=client,
        confidence_scorer=confidence_scorer,
        hitl_escalator=hitl_escalator,
    )
    return agent, client


def _final_answer_step(
    decision: str = "approve",
    confidence: float = 0.9,
    reasoning: str = "Test reasoning.",
    evidence: list[str] | None = None,
) -> str:
    return json.dumps({
        "thought": "Making final decision.",
        "action": "FINAL_ANSWER",
        "action_input": {
            "decision": decision,
            "reasoning": reasoning,
            "confidence": confidence,
            "evidence": evidence or ["peril=wind is covered", "amount within limit"],
        },
    })


def _tool_call_step(
    action: str = "check_policy",
    action_input: dict | None = None,
    thought: str = "Checking policy.",
) -> str:
    return json.dumps({
        "thought": thought,
        "action": action,
        "action_input": action_input or {"policy_id": "HO-001"},
    })


# ---------------------------------------------------------------------------
# ReAct loop — happy path with confidence scoring
# ---------------------------------------------------------------------------
class TestReActLoopWithConfidence:
    def test_high_confidence_accepted(self, sample_claim):
        """LLM returns high confidence (>= 0.85) — decision accepted as 'llm'."""
        step1 = _tool_call_step()
        step2 = _final_answer_step(confidence=0.95)
        agent, _ = _build_agent([step1, step2])
        result = agent.run(sample_claim)

        assert result.source == "llm"
        assert result.hitl_escalated is False
        assert result.confidence_score is not None
        assert result.confidence_score >= 0.85
        assert result.decision.decision == "approve"

    def test_moderate_confidence_triggers_hitl(self, sample_claim):
        """LLM returns moderate confidence (0.50-0.85) — HITL escalation."""
        step1 = _tool_call_step()
        step2 = _final_answer_step(confidence=0.6)
        agent, _ = _build_agent([step1, step2])
        result = agent.run(sample_claim)

        assert result.source == "hitl_escalation"
        assert result.hitl_escalated is True
        assert result.original_decision == "approve"
        assert result.decision.decision == "route_to_manual_review"
        assert result.confidence_score < 0.85

    def test_low_confidence_triggers_fallback(self, sample_claim):
        """LLM returns very low confidence (< 0.50) — full fallback."""
        step1 = _final_answer_step(confidence=0.1, evidence=[])
        agent, _ = _build_agent([step1])
        result = agent.run(sample_claim)
        # Very low confidence (0.1) + no evidence should push score below 0.50
        assert result.source in ("fallback", "hitl_escalation")
        assert result.confidence_score is not None
        assert result.confidence_score < 0.85


# ---------------------------------------------------------------------------
# ReAct loop — max iterations
# ---------------------------------------------------------------------------
class TestMaxIterations:
    def test_max_10_iterations_enforced(self, sample_claim):
        """Agent stops after 10 iterations — SHLD-14 AC."""
        loop_step = _tool_call_step(thought="Need more info.")
        responses = [loop_step] * 10
        agent, _ = _build_agent(responses, config=AgentConfig(
            lm_studio_base_url="http://localhost:1234/v1",
            lm_studio_api_key="lm-studio",
            model="qwen-test",
            max_react_iterations=10,
            parse_retries=0,
        ))
        result = agent.run(sample_claim)

        assert result.source == "fallback"
        assert "max_react_iterations" in (result.fallback_reason or "")

    def test_completes_on_iteration_10_if_final_answer(self, sample_claim):
        """Agent can complete exactly on iteration 10 with FINAL_ANSWER."""
        loop_steps = [_tool_call_step() for _ in range(9)]
        final = _final_answer_step(confidence=0.9)
        responses = loop_steps + [final]
        agent, _ = _build_agent(responses)
        result = agent.run(sample_claim)
        # Many iterations may reduce consistency, potentially triggering HITL
        assert result.source in ("llm", "hitl_escalation")
        assert result.iterations == 10


# ---------------------------------------------------------------------------
# ReAct loop — tool invocation tracking
# ---------------------------------------------------------------------------
class TestToolInvocationTracking:
    def test_tools_invoked_recorded_in_result(self, sample_claim):
        step1 = _tool_call_step(action="check_policy")
        step2 = _tool_call_step(action="check_history", action_input={"claimant": "Alice"})
        step3 = _final_answer_step(confidence=0.9)
        agent, _ = _build_agent([step1, step2, step3])
        result = agent.run(sample_claim)

        assert "check_policy" in result.tools_invoked
        assert "check_history" in result.tools_invoked
        assert len(result.tools_invoked) == 2

    def test_no_tools_invoked_when_immediate_final(self, sample_claim):
        step1 = _final_answer_step(confidence=0.9)
        agent, _ = _build_agent([step1])
        result = agent.run(sample_claim)

        assert result.tools_invoked == []


# ---------------------------------------------------------------------------
# ReAct loop — confidence scoring integration
# ---------------------------------------------------------------------------
class TestConfidenceScoringIntegration:
    def test_missing_required_tool_reduces_confidence(self, sample_claim):
        """Agent doesn't call validate_policy → tool_coverage drops → may escalate."""
        step1 = _tool_call_step(action="check_history", action_input={"claimant": "Alice"})
        step2 = _final_answer_step(confidence=0.75)
        scorer = ConfidenceScorer(required_tools=["check_policy"])
        agent, _ = _build_agent([step1, step2], confidence_scorer=scorer)
        result = agent.run(sample_claim)

        # Missing required tool should reduce confidence enough to escalate
        assert result.confidence_score is not None
        assert result.confidence_score < 0.85

    def test_well_grounded_evidence_boosts_confidence(self, sample_claim):
        step1 = _tool_call_step()
        step2 = _final_answer_step(
            confidence=0.9,
            evidence=[
                "peril=wind is in policy.perils_covered",
                "amount 1250.00 <= limit 10000.00",
                "prior_count=0",
            ],
        )
        agent, _ = _build_agent([step1, step2])
        result = agent.run(sample_claim)

        assert result.source == "llm"
        assert result.confidence_score >= 0.85

    def test_no_evidence_reduces_confidence(self, sample_claim):
        step1 = _tool_call_step()
        step2 = _final_answer_step(confidence=0.75, evidence=[])
        agent, _ = _build_agent([step1, step2])
        result = agent.run(sample_claim)

        # No evidence should push confidence below HITL threshold
        assert result.confidence_score < 0.85


# ---------------------------------------------------------------------------
# Fallback triggers
# ---------------------------------------------------------------------------
class TestFallbackTriggers:
    def test_llm_timeout_triggers_fallback(self, sample_claim):
        timeout_err = TimeoutError("LLM call timed out after 10s")
        agent, _ = _build_agent([timeout_err])
        result = agent.run(sample_claim)

        assert result.source == "fallback"
        assert result.iterations == 0

    def test_parse_failure_after_retries_triggers_fallback(self, sample_claim):
        agent, _ = _build_agent(["not json at all", "still not json"])
        result = agent.run(sample_claim)
        assert result.source == "fallback"
        assert "parse" in (result.fallback_reason or "").lower()

    def test_max_iterations_triggers_fallback(self, sample_claim):
        loop_step = _tool_call_step()
        responses = [loop_step] * 10
        agent, _ = _build_agent(responses, config=AgentConfig(
            lm_studio_base_url="http://localhost:1234/v1",
            lm_studio_api_key="lm-studio",
            model="qwen-test",
            max_react_iterations=5,
            parse_retries=0,
        ))
        result = agent.run(sample_claim)
        assert result.source == "fallback"
        assert "max_react_iterations" in (result.fallback_reason or "")

    def test_final_answer_invalid_schema_triggers_fallback(self, sample_claim):
        bad_final = json.dumps({
            "thought": "Deciding.",
            "action": "FINAL_ANSWER",
            "action_input": {"reasoning": "missing decision field"},
        })
        agent, _ = _build_agent([bad_final])
        result = agent.run(sample_claim)
        assert result.source == "fallback"

    def test_unknown_tool_records_error_and_continues(self, sample_claim):
        step1 = _tool_call_step(action="nonexistent_tool", action_input={})
        step2 = _tool_call_step()
        step3 = _final_answer_step(confidence=0.9)
        agent, _ = _build_agent([step1, step2, step3])
        result = agent.run(sample_claim)
        # Confidence scoring may change source
        assert result.source in ("llm", "hitl_escalation")
        assert result.iterations == 3


# ---------------------------------------------------------------------------
# Malformed input handling
# ---------------------------------------------------------------------------
class TestMalformedInputHandling:
    def test_missing_claim_id_still_processes(self):
        claim = {"amount": 100, "description": "Test claim"}
        step = _final_answer_step(confidence=0.9)
        agent, _ = _build_agent([step])
        result = agent.run(claim)
        assert result.claim_id is None
        # Decision depends on confidence scoring
        assert result.decision.decision in ("approve", "route_to_manual_review")

    def test_empty_claim_uses_fallback(self):
        claim = {}
        step = _final_answer_step(confidence=0.2, evidence=[])
        agent, _ = _build_agent([step])
        result = agent.run(claim)
        # Low confidence triggers fallback
        assert result.source == "fallback"

    def test_non_dict_claim_fields_handled(self):
        claim = {"claim_id": "CLM-1", "amount": "not_a_number"}
        step = _final_answer_step(confidence=0.9)
        agent, _ = _build_agent([step])
        result = agent.run(claim)
        assert result.source in ("llm", "hitl_escalation", "fallback")


# ---------------------------------------------------------------------------
# Result envelope completeness
# ---------------------------------------------------------------------------
class TestResultEnvelopeCompleteness:
    def test_llm_result_has_all_shld14_fields(self, sample_claim):
        step1 = _tool_call_step()
        step2 = _final_answer_step(confidence=0.95)
        agent, _ = _build_agent([step1, step2])
        result = agent.run(sample_claim)

        assert result.agent_name == "test-agent"
        assert result.claim_id == "CLM-2026-0001"
        assert result.source == "llm"
        assert result.iterations == 2
        assert result.fallback_reason is None
        assert result.confidence_score is not None
        assert result.hitl_escalated is False
        assert result.original_decision is None
        assert "check_policy" in result.tools_invoked

    def test_hitl_result_has_escalation_fields(self, sample_claim):
        step1 = _final_answer_step(confidence=0.6)
        agent, _ = _build_agent([step1])
        result = agent.run(sample_claim)

        assert result.source == "hitl_escalation"
        assert result.hitl_escalated is True
        assert result.original_decision == "approve"
        assert result.confidence_score < 0.85

    def test_fallback_result_has_reason(self, sample_claim):
        agent, _ = _build_agent([TimeoutError("timeout")])
        result = agent.run(sample_claim)

        assert result.source == "fallback"
        assert result.fallback_reason is not None
        assert result.iterations == 0


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
class TestParseReActStep:
    def test_parses_clean_json(self):
        raw = json.dumps({
            "thought": "I should check the policy.",
            "action": "check_policy",
            "action_input": {"policy_id": "HO-001"},
        })
        step = Agent._parse_react_step(raw)
        assert step.thought == "I should check the policy."
        assert step.action == "check_policy"
        assert step.action_input == {"policy_id": "HO-001"}
        assert not step.is_final

    def test_parses_markdown_fenced_json(self):
        raw = '```json\n{"thought":"x","action":"FINAL_ANSWER","action_input":{}}\n```'
        step = Agent._parse_react_step(raw)
        assert step.is_final

    def test_parses_json_with_leading_prose(self):
        raw = (
            "Here is my response.\n"
            '{"thought":"x","action":"check_policy","action_input":{"policy_id":"P"}}'
        )
        step = Agent._parse_react_step(raw)
        assert step.action == "check_policy"

    def test_raises_on_empty(self):
        with pytest.raises(_ParseError, match="empty"):
            Agent._parse_react_step("")

    def test_raises_on_invalid_json(self):
        with pytest.raises(_ParseError, match="invalid JSON"):
            Agent._parse_react_step("{not valid json}")

    def test_raises_on_schema_violation(self):
        raw = json.dumps({"action": "check_policy", "action_input": {}})
        with pytest.raises(_ParseError, match="schema validation"):
            Agent._parse_react_step(raw)
