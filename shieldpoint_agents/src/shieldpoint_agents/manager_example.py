"""
CLI demo — run a sample claim through the ManagerAgent.

Usage::

    python -m shieldpoint_agents.manager_example

Demonstrates the multi-agent orchestration:
1. ManagerAgent classifies the claim type (property, liability, etc.)
2. Produces an OrchestrationPlan
3. Invokes specialist agents (ClaimsAgent, FinancialAgent, SentimentAgent)
   in the determined sequence (sequential or parallel)
4. Synthesises their outputs into a unified ClaimDecision
5. Records every specialist output to the episodic memory store

Specialists use the FallbackEngine (no live LM Studio required) so the
demo always produces a deterministic decision.
"""

from __future__ import annotations

import json
import logging
import sys

from . import (
    AgentConfig,
    ClaimsAgent,
    ConflictResolver,
    FinancialAgent,
    InMemoryEpisodicMemory,
    ManagerAgent,
    SentimentAgent,
    build_specialists,
)

SAMPLE_CLAIMS = [
    {
        "claim_id": "CLM-MGR-DEMO-001",
        "adjuster_id": "ADJ-DEMO",
        "policy_id": "HO-2024-001",
        "claimant": "Demo Homeowner",
        "amount": 1_250.00,
        "description": "Wind damage to roof shingles during a storm.",
        "date_of_loss": "2026-03-14",
        "session_id": "sess-demo-001",
    },
    {
        "claim_id": "CLM-MGR-DEMO-002",
        "adjuster_id": "ADJ-DEMO",
        "policy_id": "AU-2024-015",
        "claimant": "Demo Driver",
        "amount": 4_800.00,
        "description": "Auto collision with bodily injury. Attorney mentioned.",
        "date_of_loss": "2026-04-02",
        "session_id": "sess-demo-002",
    },
    {
        "claim_id": "CLM-MGR-DEMO-003",
        "adjuster_id": "ADJ-DEMO",
        "policy_id": "HO-2024-012",
        "claimant": "Demo Property",
        "amount": 12_500.00,
        "description": "Flood damage. Intentional misrepresentation suspected.",
        "date_of_loss": "2026-02-28",
        "session_id": "sess-demo-003",
    },
]


def _build_manager() -> ManagerAgent:
    """Build a ManagerAgent with specialists that fall back to rule-based."""
    cfg = AgentConfig.from_env()
    specialists = build_specialists(config=cfg)  # no LLM client → fallback
    return ManagerAgent(
        config=cfg,
        specialists=specialists,
        memory=InMemoryEpisodicMemory(),
        conflict_resolver=ConflictResolver(strategy="weighted"),
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    manager = _build_manager()

    print("\n" + "=" * 72)
    print("ShieldPoint ManagerAgent — Multi-Agent Orchestration Demo")
    print("=" * 72 + "\n")

    for claim in SAMPLE_CLAIMS:
        print(f"\n>>> Processing claim {claim['claim_id']}...")
        print(f"    Description: {claim['description']}")
        result = manager.run(claim)

        print(f"\n    Claim type:        {result.plan.claim_type}")
        print(f"    Stages:            {len(result.plan.stages)}")
        for i, stage in enumerate(result.plan.stages, 1):
            print(
                f"      Stage {i} ({stage.mode}): "
                f"{', '.join(stage.agent_names)}"
            )
        print(f"    Routing rationale: {result.plan.routing_rationale}")
        print(f"    Conflict strategy: {result.plan.conflict_strategy}")
        print(f"\n    Invocations:       {len(result.invocations)}")
        for inv in result.invocations:
            conf_str = (
                f"{inv.result.confidence_score:.2f}"
                if inv.result.confidence_score is not None
                else "n/a"
            )
            print(
                f"      {inv.agent_name} ({inv.stage_id}, "
                f"{inv.duration_sec * 1000:.0f}ms): "
                f"{inv.decision_label} "
                f"(source={inv.result.source}, conf={conf_str})"
            )
        print(f"\n    Conflicts:         {len(result.conflicts)}")
        for c in result.conflicts:
            print(f"      {c.conflict_id}: {c.description}")
            print(f"        strategy={c.strategy_used}, resolution={c.resolution}")
            print(f"        rationale: {c.resolution_rationale}")
        print(f"\n    FINAL DECISION:    {result.decision.decision}")
        print(f"    Confidence:        {result.decision.confidence:.3f}")
        print(f"    Source:            {result.source}")
        print(f"    Memory episodes:   {len(result.memory_entries_used)} prior")
        print(f"    Trace id:          {result.trace_id or '(tracing disabled)'}")
        print(f"    Reasoning:         {result.decision.reasoning[:200]}...")

        # Show the episodic memory state for this claim
        episodes = manager.memory.recall(claim["claim_id"])
        print(f"\n    Episodic memory for {claim['claim_id']}: "
              f"{len(episodes)} episode(s) recorded")
        for ep in episodes:
            print(f"      {ep.episode_id} ({ep.agent_name}): "
                  f"{ep.decision_label} conf={ep.confidence:.2f}")

        print("\n" + "-" * 72)

    print("\nDemo complete. All claims processed through the ManagerAgent.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
