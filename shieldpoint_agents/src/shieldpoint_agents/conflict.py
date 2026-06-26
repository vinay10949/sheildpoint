"""
Conflict resolution for the ManagerAgent (SHLD-15).

When specialists disagree (e.g. SentimentAgent flags urgency via an
``approve`` recommendation because the description sounds calm and
cooperative, but FinancialAgent returns ``deny`` because the claim
exceeds the policy coverage limit), the ManagerAgent must synthesise a
single unified :class:`ClaimDecision` with an auditable rationale.

This module provides:

- :class:`ConflictDetector` — inspects a set of specialist invocations
  and decides whether a conflict exists (i.e. the specialists returned
  *different* decision labels).
- :class:`ConflictResolver` — given a conflict, applies one of four
  configurable strategies (``priority``, ``vote``, ``escalation``,
  ``weighted``) and returns the synthesised decision + rationale.

Strategies
----------

1. **priority** — a per-agent priority map decides the winner. The
   highest-priority agent's decision is adopted as-is. Used when one
   specialist is authoritative for a given claim dimension (e.g.
   FinancialAgent is authoritative on coverage questions).

2. **vote** — majority vote across specialists. Ties fall back to the
   ``priority`` strategy using ``tiebreak_priority_map``.

3. **escalation** — *always* route to manual review when a conflict is
   detected. Confidence is dropped to a configurable threshold; the
   final decision is ``route_to_manual_review``.

4. **weighted** — each specialist's vote is weighted by its confidence
   score. The decision label with the highest aggregate weighted
   confidence wins. This is the default strategy because it factors in
   how sure each agent is.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .manager_schemas import (
    AgentInvocationRecord,
    ConflictRecord,
)
from .schemas import ClaimDecision

logger = logging.getLogger("shieldpoint_agents.conflict")


# ---------------------------------------------------------------------------
# Default priority map — higher number = higher priority
# ---------------------------------------------------------------------------
DEFAULT_PRIORITY_MAP: dict[str, int] = {
    "FinancialAgent": 100,  # authoritative on coverage/limits
    "ClaimsAgent": 80,      # authoritative on policy validation
    "SentimentAgent": 40,   # advisory only — never authoritative alone
}


# ---------------------------------------------------------------------------
# Conflict detector
# ---------------------------------------------------------------------------
@dataclass
class ConflictDetector:
    """Detect whether specialists disagree on a claim.

    A conflict is defined as: at least two specialists returned different
    decision labels (``approve`` / ``deny`` / ``route_to_manual_review``).
    Specialists in an error state (``decision_label == 'error'``) are
    excluded from the comparison — they are handled separately by the
    ManagerAgent's fallback path.
    """

    def detect(
        self, invocations: list[AgentInvocationRecord]
    ) -> Optional[tuple[set[str], dict[str, str]]]:
        """Inspect invocations; return ``(dissenters, all_decisions)`` if conflict.

        ``all_decisions`` is a map of ``agent_name → decision_label`` for
        ALL usable specialists that participated (not just the
        dissenters). This is the map that gets embedded in the audit
        :class:`ConflictRecord` so the auditor sees the full picture.

        ``dissenters`` is the subset of agent names whose decision
        differs from the majority. Used internally by the strategies.

        Returns ``None`` when there is no conflict (or fewer than 2
        usable invocations).
        """
        usable = [i for i in invocations if i.decision_label != "error"]
        if len(usable) < 2:
            return None

        labels = {i.agent_name: i.decision_label for i in usable}
        unique_labels = set(labels.values())
        if len(unique_labels) <= 1:
            return None

        # Find the agents that are *not* in the majority group — these
        # are the dissenting agents we want to highlight in the record.
        # If every label is unique, all agents are dissenting.
        majority_label = self._majority_label(list(labels.values()))
        if majority_label is None:
            # All unique — every agent disagrees
            dissenters = set(labels.keys())
        else:
            dissenters = {
                name for name, lbl in labels.items() if lbl != majority_label
            }
        # Return the FULL labels map so the ConflictRecord has ≥2 entries.
        return dissenters, dict(labels)

    @staticmethod
    def _majority_label(labels: list[str]) -> Optional[str]:
        """Return the most common label, or None if all unique / tie."""
        counts: dict[str, int] = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        if not counts:
            return None
        max_count = max(counts.values())
        winners = [lbl for lbl, c in counts.items() if c == max_count]
        if len(winners) == 1 and max_count > 1:
            return winners[0]
        return None


# ---------------------------------------------------------------------------
# Conflict resolver
# ---------------------------------------------------------------------------
@dataclass
class ConflictResolution:
    """Result of resolving one conflict."""

    decision: ClaimDecision
    strategy: str
    rationale: str
    winning_agent: Optional[str] = None
    record: Optional[ConflictRecord] = None


@dataclass
class ConflictResolver:
    """Apply a configurable strategy to resolve specialist disagreements.

    Parameters
    ----------
    strategy : ``"priority"`` | ``"vote"`` | ``"escalation"`` | ``"weighted"``
        Default resolution strategy. Can be overridden per-call.
    priority_map : dict[str, int]
        Agent name → priority (higher wins). Used by the ``priority``
        strategy and as a tiebreaker for ``vote``.
    escalation_confidence : float
        Confidence assigned to the synthesised decision when the
        ``escalation`` strategy is used (default 0.4 — below HITL
        threshold, so the claim is routed to manual review).
    """

    strategy: str = "weighted"
    priority_map: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_PRIORITY_MAP)
    )
    escalation_confidence: float = 0.40

    # ------------------------------------------------------------------ #
    #  Public entry point                                                #
    # ------------------------------------------------------------------ #
    def resolve(
        self,
        *,
        invocations: list[AgentInvocationRecord],
        strategy: Optional[str] = None,
        claim_id: Optional[str] = None,
    ) -> ConflictResolution:
        """Resolve disagreements among ``invocations``.

        If there is no actual conflict (all agents agree, or fewer than
        two usable invocations), the resolver still returns a
        :class:`ConflictResolution` containing the agreed decision and
        a rationale noting that no conflict existed.
        """
        usable = [i for i in invocations if i.decision_label != "error"]
        if not usable:
            # All specialists errored — escalate
            return self._escalation_resolution(
                invocations=invocations,
                claim_id=claim_id,
                rationale="All specialist agents errored — escalating to manual review.",
            )

        detector = ConflictDetector()
        detected = detector.detect(invocations)
        chosen_strategy = strategy or self.strategy

        if detected is None:
            # No conflict — adopt the agreed decision with averaged confidence
            agreed_label = usable[0].decision_label
            agreed_decision = self._merge_decisions(usable, agreed_label)
            return ConflictResolution(
                decision=agreed_decision,
                strategy=chosen_strategy,
                rationale=(
                    f"No conflict — all {len(usable)} specialist(s) agreed "
                    f"on '{agreed_label}'."
                ),
                winning_agent=usable[0].agent_name,
                record=None,
            )

        dissenters, decisions_map = detected

        # Dispatch to the chosen strategy
        if chosen_strategy == "priority":
            res = self._priority_resolution(usable, dissenters, decisions_map)
        elif chosen_strategy == "vote":
            res = self._vote_resolution(usable, dissenters, decisions_map)
        elif chosen_strategy == "escalation":
            res = self._escalation_resolution(
                invocations=invocations, claim_id=claim_id,
                rationale=(
                    "Conflict detected and 'escalation' strategy is configured "
                    "— routing to manual review without further synthesis."
                ),
            )
        elif chosen_strategy == "weighted":
            res = self._weighted_resolution(usable, dissenters, decisions_map)
        else:
            logger.warning(
                "Unknown conflict strategy %r — falling back to 'weighted'", chosen_strategy,
            )
            res = self._weighted_resolution(usable, dissenters, decisions_map)

        # Build the audit ConflictRecord
        record = ConflictRecord(
            conflict_id=f"cf-{uuid.uuid4().hex[:12]}",
            agent_names=list(decisions_map.keys()),
            decisions=decisions_map,
            description=self._describe_conflict(decisions_map),
            strategy_used=chosen_strategy,
            resolution=res.decision.decision,
            resolution_rationale=res.rationale,
        )
        res.record = record
        return res

    # ------------------------------------------------------------------ #
    #  Strategies                                                        #
    # ------------------------------------------------------------------ #
    def _priority_resolution(
        self,
        usable: list[AgentInvocationRecord],
        dissenters: set[str],
        decisions_map: dict[str, str],
    ) -> ConflictResolution:
        winner_inv = max(
            usable,
            key=lambda i: self.priority_map.get(i.agent_name, 0),
        )
        winning_label = winner_inv.decision_label
        rationale = (
            f"Priority strategy: '{winner_inv.agent_name}' has the highest "
            f"priority ({self.priority_map.get(winner_inv.agent_name, 0)}) "
            f"among dissenting agents "
            f"({sorted(dissenters)}). Adopting '{winning_label}'."
        )
        decision = self._merge_decisions(usable, winning_label, prefer=winner_inv)
        return ConflictResolution(
            decision=decision,
            strategy="priority",
            rationale=rationale,
            winning_agent=winner_inv.agent_name,
        )

    def _vote_resolution(
        self,
        usable: list[AgentInvocationRecord],
        dissenters: set[str],
        decisions_map: dict[str, str],
    ) -> ConflictResolution:
        counts: dict[str, int] = {}
        for inv in usable:
            counts[inv.decision_label] = counts.get(inv.decision_label, 0) + 1
        max_count = max(counts.values())
        winners = [lbl for lbl, c in counts.items() if c == max_count]

        if len(winners) == 1:
            winning_label = winners[0]
            rationale = (
                f"Vote strategy: '{winning_label}' won with {max_count} of "
                f"{len(usable)} votes ({counts})."
            )
            # Pick the first invocation matching the winning label as the
            # 'preferred' source for evidence merge.
            preferred = next(
                inv for inv in usable if inv.decision_label == winning_label
            )
        else:
            # Tie — fall back to priority among the tied labels
            winning_label = max(
                winners,
                key=lambda lbl: max(
                    (
                        self.priority_map.get(inv.agent_name, 0)
                        for inv in usable if inv.decision_label == lbl
                    ),
                    default=0,
                ),
            )
            rationale = (
                f"Vote strategy: tie between {winners}; broke tie by priority "
                f"→ '{winning_label}'."
            )
            preferred = next(
                inv for inv in usable if inv.decision_label == winning_label
            )

        decision = self._merge_decisions(usable, winning_label, prefer=preferred)
        return ConflictResolution(
            decision=decision,
            strategy="vote",
            rationale=rationale,
            winning_agent=preferred.agent_name,
        )

    def _escalation_resolution(
        self,
        *,
        invocations: list[AgentInvocationRecord],
        claim_id: Optional[str],
        rationale: str,
    ) -> ConflictResolution:
        decision = ClaimDecision(
            decision="route_to_manual_review",
            reasoning=(
                f"Conflict among specialists could not be auto-resolved — "
                f"escalating to human review. {rationale}"
            ),
            confidence=self.escalation_confidence,
            evidence=self._collect_evidence(invocations),
        )
        return ConflictResolution(
            decision=decision,
            strategy="escalation",
            rationale=rationale,
            winning_agent=None,
        )

    def _weighted_resolution(
        self,
        usable: list[AgentInvocationRecord],
        dissenters: set[str],
        decisions_map: dict[str, str],
    ) -> ConflictResolution:
        # Weight each agent's vote by its confidence score
        weighted: dict[str, float] = {}
        for inv in usable:
            conf = float(inv.result.confidence_score or 0.0)
            weighted[inv.decision_label] = weighted.get(inv.decision_label, 0.0) + conf

        winning_label = max(weighted, key=weighted.get)
        preferred = next(
            inv for inv in usable if inv.decision_label == winning_label
        )
        rationale = (
            f"Weighted strategy: votes weighted by per-agent confidence. "
            f"Totals per label: {self._fmt(weighted)}. Adopting "
            f"'{winning_label}' (preferred agent: '{preferred.agent_name}')."
        )
        decision = self._merge_decisions(usable, winning_label, prefer=preferred)
        return ConflictResolution(
            decision=decision,
            strategy="weighted",
            rationale=rationale,
            winning_agent=preferred.agent_name,
        )

    # ------------------------------------------------------------------ #
    #  Decision-merge helpers                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge_decisions(
        invocations: list[AgentInvocationRecord],
        winning_label: str,
        *,
        prefer: Optional[AgentInvocationRecord] = None,
    ) -> ClaimDecision:
        """Build a synthesised :class:`ClaimDecision`.

        - ``decision`` = ``winning_label``
        - ``reasoning`` = the preferred agent's reasoning (or the first
          matching invocation), annotated with a synthesis note
        - ``confidence`` = average of the agents that voted for the
          winning label (so we don't artificially inflate confidence
          when a high-confidence dissenter disagreed)
        - ``evidence`` = union of all agents' evidence (deduplicated)
        """
        matching = [
            i for i in invocations if i.decision_label == winning_label
        ]
        preferred = prefer or (matching[0] if matching else invocations[0])

        confs = [
            float(i.result.confidence_score or 0.0)
            for i in matching
        ]
        avg_conf = sum(confs) / len(confs) if confs else 0.5
        # Penalise when there were dissenters
        n_dissenters = len(invocations) - len(matching)
        if n_dissenters > 0:
            avg_conf *= max(0.0, 1.0 - 0.15 * n_dissenters)
        avg_conf = max(0.0, min(1.0, avg_conf))

        evidence: list[str] = []
        seen: set[str] = set()
        for inv in invocations:
            for ev in inv.result.decision.evidence:
                if ev not in seen:
                    seen.add(ev)
                    evidence.append(ev)

        reasoning = preferred.result.decision.reasoning
        if n_dissenters > 0:
            reasoning = (
                f"[Synthesised by ManagerAgent — {n_dissenters} dissenter(s) "
                f"overruled via {preferred.agent_name}.] {reasoning}"
            )

        return ClaimDecision(
            decision=winning_label,  # type: ignore[arg-type]
            reasoning=reasoning,
            confidence=avg_conf,
            evidence=evidence,
        )

    @staticmethod
    def _collect_evidence(
        invocations: list[AgentInvocationRecord],
    ) -> list[str]:
        evidence: list[str] = []
        seen: set[str] = set()
        for inv in invocations:
            if inv.error is not None:
                continue
            for ev in inv.result.decision.evidence:
                if ev not in seen:
                    seen.add(ev)
                    evidence.append(ev)
        return evidence

    @staticmethod
    def _describe_conflict(decisions_map: dict[str, str]) -> str:
        parts = [f"{name}→{lbl}" for name, lbl in decisions_map.items()]
        return (
            "Specialists disagreed on the disposition: "
            + ", ".join(parts)
            + "."
        )

    @staticmethod
    def _fmt(weighted: dict[str, float]) -> str:
        return ", ".join(f"{lbl}={v:.2f}" for lbl, v in weighted.items())
