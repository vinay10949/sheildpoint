"""Unit tests for the Agent class — ReAct loop, parsing, fallback triggers (SHLD-14)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from shieldpoint_agents import (
    Agent,
    AgentConfig,
    FallbackEngine,
    LangfuseTracer,
    ToolRegistry,
)
from shieldpoint_agents._testing import FakeLMClient
from shieldpoint_agents.agent import _FallbackSignal, _ParseError

# ---------------------------------------------------------------------------
# Helpers — build a minimal agent with a mock LLM client and a no-op tool.
# ---------------------------------------------------------------------------
def _build_agent(
    responses: list[Any],
    *,
    config: AgentConfig | None = None,
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

    agent = Agent(
        name="test-agent",
        tools=registry,
        tracer=LangfuseTracer(agent_name="test-agent"),
        fallback=FallbackEngine(),
        config=cfg,
        llm_client=client,
    )
    return agent, client


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
        # Missing required `thought` field
        raw = json.dumps({"action": "check_policy", "action_input": {}})
        with pytest.raises(_ParseError, match="schema validation"):
            Agent._parse_react_step(raw)


# ---------------------------------------------------------------------------
# ReAct loop — happy path (with confidence scoring)
# ---------------------------------------------------------------------------
class TestReActLoopHappyPath:
    def test_full_cycle_with_one_tool_call_then_final_answer(self, sample_claim):
        step1 = json.dumps({
            "thought": "I need to verify the policy covers wind damage.",
            "action": "check_policy",
            "action_input": {"policy_id": "HO-2024-001"},
        })
        step2 = json.dumps({
            "thought": "Policy covers wind, amount is within limit. Approving.",
            "action": "FINAL_ANSWER",
            "action_input": {
                "decision": "approve",
                "reasoning": "Wind peril is covered and amount $1,250 < limit $10,000.",
                "confidence": 0.95,
                "evidence": ["peril=wind is covered", "amount within limit"],
            },
        })
        agent, client = _build_agent([step1, step2])
        result = agent.run(sample_claim)

        # With confidence 0.95 and good evidence, should be accepted as 'llm'
        assert result.source == "llm"
        assert result.iterations == 2
        assert result.decision.decision == "approve"
        assert "wind" in result.decision.reasoning.lower()
        assert result.fallback_reason is None
        assert result.confidence_score >= 0.85

    def test_immediate_final_answer_with_high_confidence(self, sample_claim):
        step1 = json.dumps({
            "thought": "This is a small claim, auto-approve.",
            "action": "FINAL_ANSWER",
            "action_input": {
                "decision": "approve",
                "reasoning": "Small claim, no red flags.",
                "confidence": 0.95,
                "evidence": ["amount within limit"],
            },
        })
        agent, client = _build_agent([step1])
        result = agent.run(sample_claim)
        assert result.iterations == 1
        # High confidence + evidence → likely accepted
        assert result.source in ("llm", "hitl_escalation")

    def test_confidence_below_hitl_triggers_escalation(self, sample_claim):
        step1 = json.dumps({
            "thought": "I need to check the policy.",
            "action": "check_policy",
            "action_input": {"policy_id": "HO-2024-001"},
        })
        step2 = json.dumps({
            "thought": "Policy covers wind but I'm not fully certain.",
            "action": "FINAL_ANSWER",
            "action_input": {
                "decision": "approve",
                "reasoning": "Wind peril may be covered but confidence is low.",
                "confidence": 0.6,
                "evidence": ["peril=wind might be covered"],
            },
        })
        agent, client = _build_agent([step1, step2])
        result = agent.run(sample_claim)

        assert result.source == "hitl_escalation"
        assert result.hitl_escalated is True
        assert result.original_decision == "approve"
        assert result.decision.decision == "route_to_manual_review"


# ---------------------------------------------------------------------------
# Fallback triggers
# ---------------------------------------------------------------------------
class TestFallbackTriggers:
    def test_llm_timeout_triggers_fallback(self, sample_claim):
        # FakeLMClient will raise on the first call.
        timeout_err = TimeoutError("LLM call timed out after 10s")
        agent, _ = _build_agent([timeout_err])
        result = agent.run(sample_claim)

        assert result.source == "fallback"
        assert "timed out" in (result.fallback_reason or "").lower() or \
               "timeout" in (result.fallback_reason or "").lower() or \
               "llm call failed" in (result.fallback_reason or "").lower()
        assert result.iterations == 0
        # The fallback should produce a valid decision for the sample claim
        # ($1,250 — between thresholds, no keywords → manual review)
        assert result.decision.decision == "route_to_manual_review"

    def test_max_iterations_triggers_fallback(self, sample_claim):
        # Provide responses, all asking for tool calls, never FINAL_ANSWER.
        loop_step = json.dumps({
            "thought": "Need more info.",
            "action": "check_policy",
            "action_input": {"policy_id": "HO-2024-001"},
        })
        responses = [loop_step] * 10
        agent, _ = _build_agent(responses, config=AgentConfig(
            lm_studio_base_url="http://localhost:1234/v1",
            lm_studio_api_key="lm-studio",
            model="qwen-test",
            max_react_iterations=3,  # tight cap
            parse_retries=0,
        ))
        result = agent.run(sample_claim)
        assert result.source == "fallback"
        assert "max_react_iterations" in (result.fallback_reason or "")

    def test_parse_failure_after_retries_triggers_fallback(self, sample_claim):
        # First response: garbage. Second response (retry): still garbage.
        agent, _ = _build_agent(["not json at all", "still not json"])
        result = agent.run(sample_claim)
        assert result.source == "fallback"
        assert "parse" in (result.fallback_reason or "").lower()

    def test_final_answer_invalid_schema_triggers_fallback(self, sample_claim):
        # FINAL_ANSWER but missing `decision` field
        bad_final = json.dumps({
            "thought": "Deciding.",
            "action": "FINAL_ANSWER",
            "action_input": {"reasoning": "missing decision field"},  # invalid
        })
        agent, _ = _build_agent([bad_final])
        result = agent.run(sample_claim)
        assert result.source == "fallback"
        assert "FINAL_ANSWER" in (result.fallback_reason or "") or \
               "ClaimDecision" in (result.fallback_reason or "")

    def test_unknown_tool_records_error_and_continues(self, sample_claim):
        step1 = json.dumps({
            "thought": "Calling a made-up tool.",
            "action": "nonexistent_tool",
            "action_input": {},
        })
        step2 = json.dumps({
            "thought": "OK, let me try the real tool.",
            "action": "check_policy",
            "action_input": {"policy_id": "HO-2024-001"},
        })
        step3 = json.dumps({
            "thought": "Got the policy info, approving.",
            "action": "FINAL_ANSWER",
            "action_input": {
                "decision": "approve",
                "reasoning": "Wind peril is covered.",
                "confidence": 0.9,
                "evidence": ["peril=wind is covered", "policy covers wind"],
            },
        })
        agent, _ = _build_agent([step1, step2, step3])
        result = agent.run(sample_claim)
        assert result.source in ("llm", "hitl_escalation")
        assert result.iterations == 3

    def test_very_low_confidence_triggers_fallback(self, sample_claim):
        """SHLD-14: confidence < 0.50 triggers rule-based fallback."""
        step1 = json.dumps({
            "thought": "I'm very uncertain about this claim.",
            "action": "FINAL_ANSWER",
            "action_input": {
                "decision": "approve",
                "reasoning": "Maybe approve, but very uncertain.",
                "confidence": 0.2,
                "evidence": [],
            },
        })
        agent, _ = _build_agent([step1])
        result = agent.run(sample_claim)
        assert result.source == "fallback"
