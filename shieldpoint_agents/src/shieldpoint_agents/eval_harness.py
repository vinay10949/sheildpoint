"""
Evaluation harness for testing the Tool-Using Agent against historical claims.

SHLD-14 AC: "Integration test: process 100 historical claims, measure accuracy
against labeled outcomes."

This module provides:
- :class:`EvaluationHarness` — runs a batch of claims through the agent and
  compares results against labeled outcomes.
- :class:`EvaluationResult` — per-claim result with match/failure details.
- :class:`EvaluationReport` — aggregate metrics (accuracy, precision, recall,
  escalation rate, etc.).
- :func:`generate_historical_claims` — generates a synthetic dataset of 100
  historical claims with labeled outcomes for testing.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent
from .claims_tools import (
    seed_claim_history,
    seed_claims,
    seed_policies,
)
from .schemas import AgentRunResult

logger = logging.getLogger("shieldpoint_agents.eval_harness")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    """Result of evaluating a single claim against its labeled outcome."""

    claim_id: str
    predicted_decision: str
    expected_decision: str
    match: bool
    source: str
    confidence_score: float | None
    iterations: int
    hitl_escalated: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "predicted_decision": self.predicted_decision,
            "expected_decision": self.expected_decision,
            "match": self.match,
            "source": self.source,
            "confidence_score": self.confidence_score,
            "iterations": self.iterations,
            "hitl_escalated": self.hitl_escalated,
            "error": self.error,
        }


@dataclass
class EvaluationReport:
    """Aggregate report from running the evaluation harness."""

    total_claims: int = 0
    correct: int = 0
    incorrect: int = 0
    errors: int = 0
    escalation_count: int = 0
    fallback_count: int = 0
    avg_confidence: float = 0.0
    avg_iterations: float = 0.0
    decision_distribution: dict[str, int] = field(default_factory=dict)
    source_distribution: dict[str, int] = field(default_factory=dict)
    per_decision_accuracy: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: list[EvaluationResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Overall accuracy = correct / (total - errors)."""
        evaluated = self.total_claims - self.errors
        return self.correct / evaluated if evaluated > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "errors": self.errors,
            "accuracy": round(self.accuracy, 4),
            "escalation_count": self.escalation_count,
            "fallback_count": self.fallback_count,
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_iterations": round(self.avg_iterations, 4),
            "decision_distribution": self.decision_distribution,
            "source_distribution": self.source_distribution,
            "per_decision_accuracy": self.per_decision_accuracy,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Human-readable summary of the evaluation."""
        lines = [
            f"Evaluation Report: {self.total_claims} claims",
            f"  Accuracy:         {self.accuracy:.1%}",
            f"  Correct:          {self.correct}",
            f"  Incorrect:        {self.incorrect}",
            f"  Errors:           {self.errors}",
            f"  HITL Escalations: {self.escalation_count}",
            f"  Fallbacks:        {self.fallback_count}",
            f"  Avg Confidence:   {self.avg_confidence:.3f}",
            f"  Avg Iterations:   {self.avg_iterations:.1f}",
            f"  Decision Distribution: {self.decision_distribution}",
            f"  Source Distribution: {self.source_distribution}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Historical claims dataset generator
# ---------------------------------------------------------------------------
# Deterministic seed for reproducibility
_RANDOM_SEED = 42

_DESCRIPTIONS_AND_OUTCOMES = [
    # (description, expected_decision, typical_amount)
    ("Wind damage to roof shingles during storm.", "approve", 1_250),
    ("Hail damage to siding and windows.", "approve", 3_200),
    ("Minor kitchen fire damage contained quickly.", "approve", 4_500),
    ("Theft of jewelry from residence.", "approve", 2_800),
    ("Vandalism to garage door.", "approve", 900),
    ("Lightning strike damaged electrical system.", "approve", 5_500),
    ("Minor water damage from pipe burst.", "approve", 1_800),
    ("Fallen tree branch damaged fence.", "approve", 650),
    ("Smoke damage from nearby fire.", "approve", 2_100),
    ("Wind damage to mailbox and garden shed.", "approve", 400),
    # Claims that should route to manual review
    ("Collision with injury reported by claimant.", "route_to_manual_review", 4_800),
    ("Slip and injury on front steps of property.", "route_to_manual_review", 3_500),
    ("Claimant has threatened litigation over claim.", "route_to_manual_review", 7_200),
    ("Attorney representing claimant for property damage.", "route_to_manual_review", 6_100),
    ("Major fire damage to entire structure.", "route_to_manual_review", 45_000),
    ("Multiple vehicle collision with injury claims.", "route_to_manual_review", 12_500),
    ("Water damage affecting multiple units in condo.", "route_to_manual_review", 15_000),
    ("Roof collapse from snow load.", "route_to_manual_review", 28_000),
    ("Mold damage discovered after water leak.", "route_to_manual_review", 8_000),
    ("Disputed liability in parking lot incident.", "route_to_manual_review", 5_200),
    # Claims that should be denied
    ("Claimant admits to intentional damage of property.", "deny", 3_000),
    ("Investigation reveals fraud in claim submission.", "deny", 10_000),
    ("Damage from flood — explicitly excluded from policy.", "deny", 20_000),
    ("Earthquake damage — not covered under policy.", "deny", 50_000),
    ("Wear and tear on 30-year-old roof.", "deny", 4_000),
]


def generate_historical_claims(
    count: int = 100,
    seed: int = _RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Generate a synthetic dataset of historical claims with labeled outcomes.

    Each claim dict includes:
    - All claim fields (claim_id, amount, description, etc.)
    - ``expected_decision`` — the labeled correct outcome.

    The dataset is deterministic (seeded) for reproducibility.
    """
    rng = random.Random(seed)
    claims: list[dict[str, Any]] = []

    policy_ids = ["HO-2024-001", "AU-2024-015", "HO-2024-088", "HO-2024-012"]
    claimants = [
        "Alice Homeowner", "Bob Driver", "Carol Resident", "Dan Property",
        "Eve Smith", "Frank Johnson", "Grace Lee", "Henry Wilson",
        "Irene Davis", "Jack Brown",
    ]

    for i in range(count):
        # Pick a description/outcome pattern, cycling through the list
        pattern_idx = i % len(_DESCRIPTIONS_AND_OUTCOMES)
        description, expected_decision, base_amount = _DESCRIPTIONS_AND_OUTCOMES[pattern_idx]

        # Add some randomness to amounts (±20%)
        amount = base_amount * rng.uniform(0.8, 1.2)

        claim_id = f"CLM-EVAL-{i + 1:04d}"
        claimant = claimants[i % len(claimants)]
        policy_id = policy_ids[i % len(policy_ids)]

        claim = {
            "claim_id": claim_id,
            "policy_id": policy_id,
            "claimant": claimant,
            "amount": round(amount, 2),
            "description": description,
            "date_of_loss": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "adjuster_id": f"ADJ-EVAL-{i % 5 + 1}",
            "expected_decision": expected_decision,
        }
        claims.append(claim)

    return claims


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------
class EvaluationHarness:
    """Run the agent against a batch of historical claims and measure accuracy.

    Parameters
    ----------
    agent : Agent
        The configured agent to evaluate.
    claims : list[dict[str, Any]]
        Historical claims with ``expected_decision`` labels.
    """

    def __init__(
        self,
        agent: Agent,
        claims: list[dict[str, Any]],
    ) -> None:
        self.agent = agent
        self.claims = claims

    def run(self) -> EvaluationReport:
        """Process all claims and build an evaluation report.

        For each claim:
        1. Run it through the agent.
        2. Compare the agent's decision against the expected decision.
        3. Record match/mismatch, confidence, source, etc.

        Returns
        -------
        EvaluationReport
            Aggregate metrics including accuracy, escalation rate, etc.
        """
        results: list[EvaluationResult] = []
        confidence_scores: list[float] = []
        iteration_counts: list[int] = []

        for claim in self.claims:
            claim_id = claim.get("claim_id", "<unknown>")
            expected = claim.get("expected_decision", "unknown")

            try:
                agent_result = self.agent.run(claim)
                predicted = agent_result.decision.decision
                match = predicted == expected
                error = None
                confidence = agent_result.confidence_score
                iterations = agent_result.iterations
                source = agent_result.source
                escalated = agent_result.hitl_escalated
            except Exception as exc:
                logger.exception("Error processing claim %s: %s", claim_id, exc)
                predicted = "error"
                match = False
                error = str(exc)
                confidence = None
                iterations = 0
                source = "error"
                escalated = False

            result = EvaluationResult(
                claim_id=claim_id,
                predicted_decision=predicted,
                expected_decision=expected,
                match=match,
                source=source,
                confidence_score=confidence,
                iterations=iterations,
                hitl_escalated=escalated,
                error=error,
            )
            results.append(result)

            if confidence is not None:
                confidence_scores.append(confidence)
            iteration_counts.append(iterations)

        # Build aggregate report
        report = EvaluationReport(
            total_claims=len(results),
            correct=sum(1 for r in results if r.match),
            incorrect=sum(1 for r in results if not r.match and r.error is None),
            errors=sum(1 for r in results if r.error is not None),
            escalation_count=sum(1 for r in results if r.hitl_escalated),
            fallback_count=sum(1 for r in results if r.source == "fallback"),
            avg_confidence=(
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores else 0.0
            ),
            avg_iterations=(
                sum(iteration_counts) / len(iteration_counts)
                if iteration_counts else 0.0
            ),
            results=results,
        )

        # Decision distribution
        decision_dist: dict[str, int] = {}
        for r in results:
            decision_dist[r.predicted_decision] = decision_dist.get(r.predicted_decision, 0) + 1
        report.decision_distribution = decision_dist

        # Source distribution
        source_dist: dict[str, int] = {}
        for r in results:
            source_dist[r.source] = source_dist.get(r.source, 0) + 1
        report.source_distribution = source_dist

        # Per-decision accuracy
        per_decision: dict[str, dict[str, Any]] = {}
        for expected_decision in set(r.expected_decision for r in results):
            subset = [r for r in results if r.expected_decision == expected_decision]
            correct = sum(1 for r in subset if r.match)
            per_decision[expected_decision] = {
                "total": len(subset),
                "correct": correct,
                "accuracy": round(correct / len(subset), 4) if subset else 0.0,
            }
        report.per_decision_accuracy = per_decision

        logger.info("Evaluation complete: %s", report.summary())
        return report
