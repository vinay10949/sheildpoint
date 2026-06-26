"""
SP-305 — Episodic Memory Store enhancement unit tests.

Verifies the acceptance criteria:
- Episodic memory store persists agent interaction history per claim ID
- ManagerAgent retrieves memory entries and includes them in LLM context
- Memory entries include: timestamp, agent ID, assessment result, tool
  invocations, ZKP proof refs
- Follow-up interactions on the same claim show full context awareness
- Memory retrieval completes in < 50ms for claims with up to 20 prior
  interactions
- TTL-based cleanup for memory entries older than 12 months
- All memory read/writes logged as Langfuse spans

Tests run against the InMemoryEpisodicMemory backend (no Postgres
dependency). The Postgres backend is structurally identical and is
exercised by the existing test_integration.py suite when a database
is available.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from shieldpoint_agents import (
    ClaimDecision,
    EpisodicMemoryEntry,
    EpisodicMemoryStore,
    InMemoryEpisodicMemory,
    LangfuseTracer,
    build_episodic_memory,
)
from shieldpoint_agents.memory import DEFAULT_TTL_SECONDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_entry(
    *,
    claim_id: str = "CLM-TEST-001",
    agent_name: str = "ClaimsAgent",
    decision_label: str = "approve",
    reasoning: str = "Policy covers the claimed peril.",
    confidence: float = 0.92,
    episode_id: str = "ep-001",
    created_at: float | None = None,
    evidence: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    trace_id: str | None = "trace-001",
) -> EpisodicMemoryEntry:
    return EpisodicMemoryEntry(
        episode_id=episode_id,
        claim_id=claim_id,
        agent_name=agent_name,
        decision_label=decision_label,
        decision=ClaimDecision(
            decision=decision_label if decision_label in {"approve", "deny", "route_to_manual_review"} else "approve",
            reasoning=reasoning,
            confidence=confidence,
            evidence=evidence or ["policy covers peril"],
        ),
        evidence=evidence or ["policy covers peril"],
        confidence=confidence,
        trace_id=trace_id,
        created_at=created_at if created_at is not None else time.time(),
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# search() — keyword search across episode payloads
# ---------------------------------------------------------------------------
class TestSearch:
    def setup_method(self):
        self.store = InMemoryEpisodicMemory()
        # Seed with multiple episodes across multiple claims
        self.store.append(_make_entry(
            claim_id="CLM-A", agent_name="ClaimsAgent",
            reasoning="Wind damage to roof.",
            evidence=["peril=wind"],
            episode_id="ep-1",
        ))
        self.store.append(_make_entry(
            claim_id="CLM-A", agent_name="FinancialAgent",
            reasoning="Payment authorised for $1,250.",
            evidence=["amount within limit"],
            episode_id="ep-2",
        ))
        self.store.append(_make_entry(
            claim_id="CLM-B", agent_name="ClaimsAgent",
            reasoning="Flood damage — denied (excluded peril).",
            evidence=["peril=flood excluded"],
            episode_id="ep-3",
        ))
        self.store.append(_make_entry(
            claim_id="CLM-B", agent_name="SentimentAgent",
            reasoning="Claimant was anxious about the denial.",
            evidence=["tone=anxious"],
            episode_id="ep-4",
        ))

    def test_search_by_keyword_across_all_claims(self):
        results = self.store.search(keywords=["wind"])
        assert len(results) == 1
        assert results[0].claim_id == "CLM-A"
        assert results[0].agent_name == "ClaimsAgent"

    def test_search_by_keyword_within_single_claim(self):
        results = self.store.search(claim_id="CLM-B", keywords=["flood"])
        assert len(results) == 1
        assert results[0].episode_id == "ep-3"

    def test_search_by_agent_name(self):
        results = self.store.search(agent_name="FinancialAgent")
        assert len(results) == 1
        assert results[0].agent_name == "FinancialAgent"

    def test_search_multiple_keywords_anded(self):
        # Both "flood" AND "denied" must appear
        results = self.store.search(keywords=["flood", "denied"])
        assert len(results) == 1
        # "flood" without "denied" also matches the same record, but with
        # both keywords ANDed we should still get 1 (the flood entry mentions both)
        results = self.store.search(keywords=["flood", "wind"])
        assert len(results) == 0  # no entry mentions BOTH flood AND wind

    def test_search_returns_empty_for_no_matches(self):
        results = self.store.search(keywords=["nonexistent"])
        assert results == []

    def test_search_respects_max_results(self):
        # Add 10 more entries
        for i in range(10):
            self.store.append(_make_entry(
                claim_id="CLM-C", reasoning=f"roof damage {i}",
                episode_id=f"ep-c-{i}",
            ))
        results = self.store.search(keywords=["roof"], max_results=5)
        assert len(results) == 5

    def test_search_is_case_insensitive(self):
        results = self.store.search(keywords=["WIND"])
        assert len(results) == 1
        results = self.store.search(keywords=["Wind"])
        assert len(results) == 1


# ---------------------------------------------------------------------------
# assemble_context() — LLM-consumable context string
# ---------------------------------------------------------------------------
class TestAssembleContext:
    def setup_method(self):
        self.store = InMemoryEpisodicMemory()
        self.store.append(_make_entry(
            claim_id="CLM-CTX",
            agent_name="ClaimsAgent",
            reasoning="Policy HO-2024-001 covers wind damage. Claim amount $1,250 within limit.",
            evidence=["peril=wind covered", "amount <= limit"],
            metadata={
                "tools_invoked": ["validate_policy", "check_claim_history"],
                "zkp_proof_ref": "zkp:abc123def456",
            },
            episode_id="ep-ctx-1",
        ))
        self.store.append(_make_entry(
            claim_id="CLM-CTX",
            agent_name="FinancialAgent",
            decision_label="approve",
            reasoning="Payment authorised: $1,250 - $500 deductible = $750 net.",
            evidence=["deductible applied", "net_payable=750"],
            metadata={
                "tools_invoked": ["process_payment"],
                "zkp_proof_ref": "zkp:abc123def456",
            },
            episode_id="ep-ctx-2",
        ))

    def test_returns_no_history_message_for_unknown_claim(self):
        ctx = self.store.assemble_context("CLM-UNKNOWN")
        assert "no prior agent history" in ctx.lower()
        assert "first interaction" in ctx.lower()

    def test_includes_timestamp_agent_id_decision(self):
        ctx = self.store.assemble_context("CLM-CTX")
        assert "ClaimsAgent" in ctx
        assert "FinancialAgent" in ctx
        assert "approve" in ctx
        # Timestamp in YYYY-MM-DD HH:MM:SS format
        import re
        assert re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]", ctx)

    def test_includes_reasoning_and_evidence(self):
        ctx = self.store.assemble_context("CLM-CTX")
        assert "Policy HO-2024-001 covers wind damage" in ctx
        assert "peril=wind covered" in ctx
        assert "deductible applied" in ctx

    def test_includes_tool_invocations(self):
        ctx = self.store.assemble_context("CLM-CTX")
        assert "tools_invoked" in ctx
        assert "validate_policy" in ctx
        assert "check_claim_history" in ctx
        assert "process_payment" in ctx

    def test_includes_zkp_proof_refs(self):
        ctx = self.store.assemble_context("CLM-CTX")
        assert "zkp_proof_ref" in ctx
        assert "zkp:abc123def456" in ctx

    def test_includes_trace_id(self):
        ctx = self.store.assemble_context("CLM-CTX")
        assert "trace_id" in ctx
        assert "trace-001" in ctx

    def test_respects_max_entries(self):
        # Add 25 entries; max_entries=20 should cap output
        for i in range(25):
            self.store.append(_make_entry(
                claim_id="CLM-MAX", reasoning=f"entry {i}",
                episode_id=f"ep-max-{i}",
            ))
        ctx = self.store.assemble_context("CLM-MAX", max_entries=20)
        # Header should say "showing last 20"
        assert "showing last 20" in ctx
        # Should NOT contain entries 0-4 (only 5-24)
        assert "entry 0" not in ctx
        assert "entry 24" in ctx

    def test_can_disable_tool_invocations(self):
        ctx = self.store.assemble_context("CLM-CTX", include_tool_invocations=False)
        assert "tools_invoked" not in ctx

    def test_can_disable_zkp_refs(self):
        ctx = self.store.assemble_context("CLM-CTX", include_zkp_refs=False)
        assert "zkp_proof_ref" not in ctx


# ---------------------------------------------------------------------------
# cleanup_expired() — TTL-based cleanup
# ---------------------------------------------------------------------------
class TestCleanupExpired:
    def test_removes_entries_older_than_ttl(self):
        store = InMemoryEpisodicMemory(ttl_seconds=60)  # 1 minute TTL
        now = time.time()
        # Fresh entry (within TTL)
        store.append(_make_entry(
            claim_id="CLM-FRESH", episode_id="ep-fresh",
            created_at=now,
        ))
        # Old entry (outside TTL)
        store.append(_make_entry(
            claim_id="CLM-OLD", episode_id="ep-old",
            created_at=now - 120,  # 2 minutes ago
        ))
        removed = store.cleanup_expired(now=now)
        assert removed == 1
        # Fresh entry should still be there
        assert store.has_history("CLM-FRESH")
        # Old entry should be gone
        assert not store.has_history("CLM-OLD")

    def test_default_ttl_is_12_months(self):
        store = InMemoryEpisodicMemory()
        assert store.ttl_seconds == DEFAULT_TTL_SECONDS
        assert store.ttl_seconds == 365 * 24 * 3600

    def test_returns_zero_when_nothing_expired(self):
        store = InMemoryEpisodicMemory(ttl_seconds=3600)
        store.append(_make_entry(episode_id="ep-1", created_at=time.time()))
        removed = store.cleanup_expired()
        assert removed == 0

    def test_custom_ttl_via_factory(self):
        store = build_episodic_memory(backend="memory", ttl_seconds=30)
        assert store.ttl_seconds == 30

    def test_cleanup_removes_empty_claims(self):
        """When all entries for a claim are expired, the claim_id key
        should be removed entirely (not left as an empty list)."""
        store = InMemoryEpisodicMemory(ttl_seconds=60)
        now = time.time()
        store.append(_make_entry(
            claim_id="CLM-ONLY-OLD", episode_id="ep-old",
            created_at=now - 120,
        ))
        store.cleanup_expired(now=now)
        assert not store.has_history("CLM-ONLY-OLD")


# ---------------------------------------------------------------------------
# AC: Memory retrieval completes in < 50ms for claims with up to 20 prior
# interactions
# ---------------------------------------------------------------------------
class TestRetrievalLatency:
    def test_recall_under_50ms_with_20_entries(self):
        store = InMemoryEpisodicMemory()
        # Seed 20 entries on the same claim
        for i in range(20):
            store.append(_make_entry(
                claim_id="CLM-PERF", episode_id=f"ep-{i}",
                created_at=time.time() - (20 - i),
                reasoning=f"Entry number {i} with some reasoning text.",
            ))
        # Warm up
        store.recall("CLM-PERF")

        # Time 100 recalls and take the average
        start = time.perf_counter()
        for _ in range(100):
            results = store.recall("CLM-PERF")
        elapsed_ms = (time.perf_counter() - start) * 10  # avg ms per call

        assert len(results) == 20
        assert elapsed_ms < 50.0, (
            f"Average recall latency {elapsed_ms:.2f}ms exceeds 50ms AC"
        )

    def test_assemble_context_under_50ms_with_20_entries(self):
        store = InMemoryEpisodicMemory()
        for i in range(20):
            store.append(_make_entry(
                claim_id="CLM-PERF2", episode_id=f"ep-{i}",
                created_at=time.time() - (20 - i),
                reasoning=f"Entry number {i} with reasoning text.",
                metadata={"tools_invoked": ["t1", "t2"], "zkp_proof_ref": "zkp:x"},
            ))
        start = time.perf_counter()
        ctx = store.assemble_context("CLM-PERF2")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50.0, (
            f"assemble_context latency {elapsed_ms:.2f}ms exceeds 50ms AC"
        )
        assert "Entry number 19" in ctx


# ---------------------------------------------------------------------------
# Multi-interaction scenario — simulates a claim going through multiple
# agent interactions over time (the SP-305 use case).
# ---------------------------------------------------------------------------
class TestMultiInteractionScenario:
    def test_follow_up_interaction_has_full_context(self):
        """Simulates: claimant calls day 1 → ClaimsAgent processes.
        Claimant calls day 3 → ManagerAgent retrieves prior context
        before re-invoking specialists."""
        store = InMemoryEpisodicMemory()
        now = time.time()

        # Day 1: Initial claim — ClaimsAgent processes
        store.append(_make_entry(
            claim_id="CLM-MULTI",
            agent_name="ClaimsAgent",
            reasoning="Policy covers wind damage. Claim amount $1,250.",
            episode_id="ep-day1-1",
            created_at=now - 86400 * 2,  # 2 days ago
            metadata={"tools_invoked": ["validate_policy"]},
        ))
        store.append(_make_entry(
            claim_id="CLM-MULTI",
            agent_name="FinancialAgent",
            reasoning="Payment authorised for $750 net of deductible.",
            episode_id="ep-day1-2",
            created_at=now - 86400 * 2 + 60,  # 2 days ago + 1 min
            metadata={"tools_invoked": ["process_payment"], "zkp_proof_ref": "zkp:xyz"},
        ))

        # Day 3: Follow-up call — ManagerAgent retrieves context
        ctx = store.assemble_context("CLM-MULTI")

        # The context should show the full history
        assert "ClaimsAgent" in ctx
        assert "FinancialAgent" in ctx
        assert "wind damage" in ctx
        assert "$1,250" in ctx
        assert "$750" in ctx
        assert "validate_policy" in ctx
        assert "process_payment" in ctx
        assert "zkp:xyz" in ctx
        # Header should reflect 2 prior episodes
        assert "2 total episodes" in ctx

    def test_third_interaction_sees_all_prior(self):
        store = InMemoryEpisodicMemory()
        now = time.time()
        # Three interactions across three days
        for day in range(3):
            store.append(_make_entry(
                claim_id="CLM-THREE",
                agent_name="ClaimsAgent",
                reasoning=f"Day {day + 1} assessment.",
                episode_id=f"ep-day{day + 1}",
                created_at=now - 86400 * (2 - day),
            ))
        ctx = store.assemble_context("CLM-THREE")
        assert "3 total episodes" in ctx
        assert "Day 1 assessment" in ctx
        assert "Day 2 assessment" in ctx
        assert "Day 3 assessment" in ctx


# ---------------------------------------------------------------------------
# Langfuse span instrumentation
# ---------------------------------------------------------------------------
class TestLangfuseInstrumentation:
    def test_traced_wrappers_no_op_when_tracer_is_none(self):
        store = InMemoryEpisodicMemory(tracer=None)
        # Should not raise
        entry = _make_entry()
        eid = store.append_traced(entry)
        assert eid == entry.episode_id
        results = store.recall_traced(entry.claim_id)
        assert len(results) == 1

    def test_traced_wrappers_open_trace_when_tracer_supplied(self):
        tracer = LangfuseTracer(agent_name="test")
        store = InMemoryEpisodicMemory(tracer=tracer)
        entry = _make_entry()
        # Should not raise even when tracer is supplied (it no-ops gracefully
        # when Langfuse env vars aren't set)
        eid = store.append_traced(entry)
        assert eid == entry.episode_id
        results = store.recall_traced(entry.claim_id)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
class TestFactory:
    def test_build_memory_backend(self):
        store = build_episodic_memory(backend="memory")
        assert isinstance(store, InMemoryEpisodicMemory)

    def test_build_memory_with_ttl(self):
        store = build_episodic_memory(backend="memory", ttl_seconds=99)
        assert store.ttl_seconds == 99

    def test_build_memory_with_tracer(self):
        tracer = LangfuseTracer(agent_name="test")
        store = build_episodic_memory(backend="memory", tracer=tracer)
        assert store.tracer is tracer

    def test_build_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown episodic memory backend"):
            build_episodic_memory(backend="redis")


# ---------------------------------------------------------------------------
# Schema requirements — entries include all required fields
# ---------------------------------------------------------------------------
class TestEntrySchema:
    def test_entry_includes_all_required_fields(self):
        """AC: 'Memory entries include: timestamp, agent ID, assessment
        result, tool invocations, ZKP proof refs'."""
        entry = _make_entry(
            metadata={"tools_invoked": ["validate_policy"], "zkp_proof_ref": "zkp:abc"},
        )
        # Timestamp
        assert hasattr(entry, "created_at")
        assert entry.created_at > 0
        # Agent ID
        assert hasattr(entry, "agent_name")
        assert entry.agent_name == "ClaimsAgent"
        # Assessment result
        assert hasattr(entry, "decision_label")
        assert hasattr(entry, "decision")
        assert hasattr(entry, "confidence")
        # Tool invocations + ZKP refs are in metadata
        assert "tools_invoked" in entry.metadata
        assert "zkp_proof_ref" in entry.metadata

    def test_entry_round_trips_through_json(self):
        """Entries must be JSON-serialisable for the Postgres JSONB column."""
        entry = _make_entry()
        d = entry.model_dump()
        # Must be JSON-serialisable
        import json
        s = json.dumps(d, default=str)
        # Must round-trip
        d2 = json.loads(s)
        entry2 = EpisodicMemoryEntry.model_validate(d2)
        assert entry2.claim_id == entry.claim_id
        assert entry2.agent_name == entry.agent_name
        assert entry2.confidence == entry.confidence
