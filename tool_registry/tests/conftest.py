"""Pytest configuration and shared fixtures for the ShieldPoint tool tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the package importable without installation.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shieldpoint import (  # noqa: E402
    ClaimStateMachine,
    InMemoryClaimsRepository,
    InMemoryPaymentLedgerRepository,
    InMemoryPolicyRepository,
    NullSpanRecorder,
    build_default_registry,
)


# ---------------------------------------------------------------------------
# Repositories — fresh per-test to keep tests isolated
# ---------------------------------------------------------------------------
@pytest.fixture
def claims_repo() -> InMemoryClaimsRepository:
    return InMemoryClaimsRepository()


@pytest.fixture
def policy_repo() -> InMemoryPolicyRepository:
    return InMemoryPolicyRepository()


@pytest.fixture
def payment_repo() -> InMemoryPaymentLedgerRepository:
    return InMemoryPaymentLedgerRepository()


@pytest.fixture
def state_machine() -> ClaimStateMachine:
    return ClaimStateMachine()


@pytest.fixture
def span_recorder() -> NullSpanRecorder:
    return NullSpanRecorder()


@pytest.fixture
def registry(
    claims_repo: InMemoryClaimsRepository,
    policy_repo: InMemoryPolicyRepository,
    payment_repo: InMemoryPaymentLedgerRepository,
    state_machine: ClaimStateMachine,
    span_recorder: NullSpanRecorder,
):
    """A registry wired up with fresh in-memory repos for every test."""
    return build_default_registry(
        claims_repo=claims_repo,
        policy_repo=policy_repo,
        payment_repo=payment_repo,
        state_machine=state_machine,
        span_recorder=span_recorder,
    )


# ---------------------------------------------------------------------------
# Mock PostgreSQL repositories — for tests that need to assert on DB calls
# without the in-memory seed data
# ---------------------------------------------------------------------------
class MockClaimsRepository:
    """Mock claims repo that records every call for assertion."""

    def __init__(self, claims: dict[str, dict] | None = None) -> None:
        self._claims = dict(claims or {})
        self.get_claim_calls: list[str] = []
        self.update_status_calls: list[tuple[str, str, str]] = []

    def get_claim(self, claim_id: str):
        self.get_claim_calls.append(claim_id)
        c = self._claims.get(claim_id)
        return dict(c) if c else None

    def update_status(self, claim_id: str, new_status: str, *, updated_by: str = "system"):
        self.update_status_calls.append((claim_id, new_status, updated_by))
        c = self._claims.get(claim_id)
        if c is None:
            return None
        c["status"] = new_status
        return dict(c)


class MockPolicyRepository:
    def __init__(self, policies: dict[str, dict] | None = None) -> None:
        self._policies = dict(policies or {})
        self.get_policy_calls: list[str] = []

    def get_policy(self, policy_id: str):
        self.get_policy_calls.append(policy_id)
        p = self._policies.get(policy_id)
        return dict(p) if p else None


class MockPaymentLedgerRepository:
    def __init__(self, records: list[dict] | None = None) -> None:
        self._records = list(records or [])
        self.insert_calls: list[dict] = []

    def find_by_claim(self, claim_id: str):
        return [dict(r) for r in self._records if r["claim_id"] == claim_id]

    def find_by_idempotency_key(self, key: str):
        for r in self._records:
            if r.get("idempotency_key") == key:
                return dict(r)
        return None

    def insert(self, record: dict):
        self.insert_calls.append(dict(record))
        record = dict(record)
        if "payment_id" not in record:
            record["payment_id"] = f"PMT-TEST-{len(self._records):04d}"
        self._records.append(record)
        return dict(record)


@pytest.fixture
def mock_claims_repo() -> MockClaimsRepository:
    return MockClaimsRepository()


@pytest.fixture
def mock_policy_repo() -> MockPolicyRepository:
    return MockPolicyRepository()


@pytest.fixture
def mock_payment_repo() -> MockPaymentLedgerRepository:
    return MockPaymentLedgerRepository()
