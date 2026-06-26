"""
CLI demo — run a sample claim through the agent framework.

Usage::

    python -m shieldpoint_agents.example

If LM Studio is running on ``$LM_STUDIO_BASE_URL``, the agent will use the
real Qwen model. If LM Studio is unreachable, the FallbackEngine kicks in
and the demo still produces a deterministic decision.
"""

from __future__ import annotations

import json
import logging
import sys

from . import (
    Agent,
    AgentConfig,
    FallbackEngine,
    LangfuseTracer,
    ToolRegistry,
)

SAMPLE_CLAIM = {
    "claim_id": "CLM-DEMO-0001",
    "adjuster_id": "ADJ-DEMO",
    "policy_id": "HO-2024-DEMO",
    "claimant": "Demo Homeowner",
    "amount": 1_250.00,
    "description": "Wind damage to roof shingles during a storm.",
    "date_of_loss": "2026-03-14",
}


def _build_demo_agent() -> Agent:
    registry = ToolRegistry()

    @registry.register(
        name="validate_policy",
        description="Look up a policy by ID and return its coverage limits.",
        schema={
            "type": "object",
            "properties": {"policy_id": {"type": "string"}},
            "required": ["policy_id"],
            "additionalProperties": False,
        },
    )
    def validate_policy(policy_id: str) -> dict:
        return {
            "policy_id": policy_id,
            "limit": 25_000,
            "deductible": 1_000,
            "perils_covered": ["wind", "hail", "fire"],
            "perils_excluded": ["flood", "earthquake"],
        }

    return Agent(
        name="claim-classifier-demo",
        tools=registry,
        tracer=LangfuseTracer(agent_name="claim-classifier-demo"),
        fallback=FallbackEngine(),
        config=AgentConfig.from_env(),
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    agent = _build_demo_agent()
    result = agent.run(SAMPLE_CLAIM)
    print(json.dumps(result.model_dump(), indent=2, default=str))
    return 0 if result.source in ("llm", "fallback") else 1


if __name__ == "__main__":
    sys.exit(main())
