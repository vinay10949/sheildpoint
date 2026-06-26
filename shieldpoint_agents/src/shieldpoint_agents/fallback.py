"""
FallbackEngine — rule-based claim processing for when the LLM is unavailable.

AC: "Graceful fallback: if LLM call fails or times out (>10s), rule-based
fallback executes and logs reason".

This module implements deterministic claim-processing rules that produce the
same :class:`ClaimDecision` shape as a successful ReAct run. The rules are
intentionally conservative — when in doubt, route to manual review.

Rules (applied in order, first match wins):

1. **Hard deny** — claim description contains any of the explicit-exclusion
   keywords (default: ``"fraud"``, ``"intentional"``). Confidence: 0.95.
2. **Auto-approve** — amount < ``auto_approve_threshold`` (default $500) AND
   no review-flagged keywords. Confidence: 0.80.
3. **Manual review** — amount >= ``manual_review_threshold`` (default $5,000)
   OR description contains any of the review-flagged keywords (default:
   ``"injury"``, ``"litigation"``, ``"attorney"``). Confidence: 0.50.
4. **Default** — route to manual review at confidence 0.40.

The thresholds and keyword lists are configurable via :class:`FallbackConfig`
so agents can be tuned per product line (home, auto, etc.) without code
changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .schemas import AgentRunResult, ClaimDecision

logger = logging.getLogger("shieldpoint_agents.fallback")


@dataclass(frozen=True)
class FallbackConfig:
    """Tunable knobs for :class:`FallbackEngine`."""

    auto_approve_threshold: float = 500.0
    manual_review_threshold: float = 5_000.0
    deny_keywords: tuple[str, ...] = ("fraud", "intentional")
    review_keywords: tuple[str, ...] = ("injury", "litigation", "attorney")
    default_confidence: float = 0.40


@dataclass
class FallbackResult:
    """Internal record produced by :meth:`FallbackEngine.evaluate`.

    Wrapped in :class:`AgentRunResult` by the caller (the Agent).
    """

    decision: ClaimDecision
    rule_name: str
    reason: str


class FallbackEngine:
    """Deterministic claim processor — used when the LLM is unavailable.

    Construct with a custom :class:`FallbackConfig` to tune thresholds; or
    call :meth:`evaluate` directly on a claim dict.
    """

    def __init__(self, config: FallbackConfig | None = None) -> None:
        self.config = config or FallbackConfig()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def evaluate(self, claim: dict[str, Any]) -> FallbackResult:
        """Apply rule-based logic and return a :class:`FallbackResult`.

        ``claim`` is expected to have at least ``amount`` (float) and
        ``description`` (str). Missing fields degrade gracefully: missing
        amount is treated as +inf (forces manual review); missing description
        is treated as the empty string.
        """
        amount = float(claim.get("amount", float("inf")))
        description = str(claim.get("description", "")).lower()
        claim_id = claim.get("claim_id")

        # Rule 1: explicit deny
        for kw in self.config.deny_keywords:
            if kw in description:
                return self._make_result(
                    rule="deny_keyword",
                    reason=(
                        f"Description contains deny keyword '{kw}' — auto-deny "
                        "per policy exclusion list."
                    ),
                    decision=ClaimDecision(
                        decision="deny",
                        reasoning=(
                            f"Claim contains '{kw}' which is on the policy "
                            "exclusion list. Auto-denied."
                        ),
                        confidence=0.95,
                        evidence=[f"deny keyword matched: '{kw}'"],
                    ),
                    claim_id=claim_id,
                )

        # Rule 2: small-amount auto-approve
        review_hit = next(
            (kw for kw in self.config.review_keywords if kw in description),
            None,
        )
        if amount < self.config.auto_approve_threshold and not review_hit:
            return self._make_result(
                rule="small_amount_auto_approve",
                reason=(
                    f"Amount ${amount:,.2f} below auto-approve threshold "
                    f"${self.config.auto_approve_threshold:,.2f} and no "
                    "review keywords matched."
                ),
                decision=ClaimDecision(
                    decision="approve",
                    reasoning=(
                        f"Claim amount ${amount:,.2f} is below the "
                        f"${self.config.auto_approve_threshold:,.2f} auto-"
                        "approve threshold and no review triggers were "
                        "detected in the description."
                    ),
                    confidence=0.80,
                    evidence=[
                        f"amount={amount:.2f} < {self.config.auto_approve_threshold:.2f}",
                        "no review keywords matched",
                    ],
                ),
                claim_id=claim_id,
            )

        # Rule 3: high-amount or review-keyword → manual review
        if amount >= self.config.manual_review_threshold or review_hit:
            triggers = []
            if amount >= self.config.manual_review_threshold:
                triggers.append(
                    f"amount ${amount:,.2f} >= "
                    f"${self.config.manual_review_threshold:,.2f}"
                )
            if review_hit:
                triggers.append(f"review keyword '{review_hit}'")
            return self._make_result(
                rule="manual_review_trigger",
                reason="; ".join(triggers),
                decision=ClaimDecision(
                    decision="route_to_manual_review",
                    reasoning=(
                        "Claim flagged for manual review: " + "; ".join(triggers)
                        + ". Adjuster should verify coverage and documentation."
                    ),
                    confidence=0.50,
                    evidence=triggers,
                ),
                claim_id=claim_id,
            )

        # Rule 4: default — manual review at low confidence
        return self._make_result(
            rule="default_manual_review",
            reason="No rule matched; defaulting to manual review.",
            decision=ClaimDecision(
                decision="route_to_manual_review",
                reasoning=(
                    "Claim did not match any auto-decision rule. Routing to "
                    "manual review for adjuster assessment."
                ),
                confidence=self.config.default_confidence,
                evidence=["no rule matched"],
            ),
            claim_id=claim_id,
        )

    # ------------------------------------------------------------------ #
    #  Convenience — produce a full AgentRunResult envelope               #
    # ------------------------------------------------------------------ #
    def run(
        self,
        claim: dict[str, Any],
        *,
        agent_name: str = "fallback-engine",
        fallback_reason: str = "llm_unavailable",
    ) -> AgentRunResult:
        """Evaluate ``claim`` and wrap the result in an :class:`AgentRunResult`.

        Used by :class:`Agent.run` when the LLM path fails. The
        ``fallback_reason`` argument is logged and surfaced in the result
        envelope for traceability.
        """
        result = self.evaluate(claim)
        logger.info(
            "Fallback engaged for claim %s: rule=%s reason=%s upstream_cause=%s",
            claim.get("claim_id", "<unknown>"),
            result.rule_name,
            result.reason,
            fallback_reason,
        )
        return AgentRunResult(
            agent_name=agent_name,
            claim_id=claim.get("claim_id"),
            decision=result.decision,
            source="fallback",
            iterations=0,
            fallback_reason=f"{fallback_reason}; rule={result.rule_name}; {result.reason}",
            trace_id=None,
        )

    # ------------------------------------------------------------------ #
    #  Internal                                                           #
    # ------------------------------------------------------------------ #
    def _make_result(
        self,
        *,
        rule: str,
        reason: str,
        decision: ClaimDecision,
        claim_id: str | None,
    ) -> FallbackResult:
        logger.debug(
            "Fallback rule '%s' fired for claim %s: %s",
            rule, claim_id or "<unknown>", reason,
        )
        return FallbackResult(
            decision=decision,
            rule_name=rule,
            reason=reason,
        )
