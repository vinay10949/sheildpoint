"""
Database repository layer for the ShieldPoint tool registry.

The tools required by SP-201 (``claim_lookup``, ``policy_validate``,
``payment_authorize``, ``claim_update_status``) all need to read or write
persistent state: claims, policies, and the payment ledger. This module
defines the abstract repository interfaces and ships in-memory
implementations suitable for unit tests and local development.

Production wiring
-----------------
Each abstract class has a concrete PostgreSQL counterpart intended for
production:

- :class:`ClaimsRepository`          → ``PostgresClaimsRepository`` (psycopg2/SQLAlchemy)
- :class:`PolicyRepository`          → ``PostgresPolicyRepository``
- :class:`PaymentLedgerRepository`   → ``PostgresPaymentLedgerRepository``

The PostgreSQL implementations are intentionally *not* included in this
package — they live in the deployment repo so this module stays
dependency-free and unit-testable without a running Postgres instance.
The docstrings on each abstract method specify the exact SQL schema the
Postgres implementation must target, so the production wiring is fully
determined by this interface.

Schema (PostgreSQL)
-------------------
.. code-block:: sql

    -- claims table
    CREATE TABLE claims (
        claim_id         TEXT PRIMARY KEY,
        policy_id        TEXT NOT NULL,
        claimant         TEXT NOT NULL,
        amount           NUMERIC(12,2) NOT NULL,
        description      TEXT,
        date_of_loss     DATE,
        status           TEXT NOT NULL,    -- one of ClaimState values
        adjuster_id      TEXT,
        documents        JSONB DEFAULT '[]',
        created_at       TIMESTAMPTZ DEFAULT now(),
        updated_at       TIMESTAMPTZ DEFAULT now()
    );

    -- policies table
    CREATE TABLE policies (
        policy_id        TEXT PRIMARY KEY,
        type             TEXT NOT NULL,
        policyholder     TEXT NOT NULL,
        limit            NUMERIC(12,2) NOT NULL,
        deductible       NUMERIC(12,2) NOT NULL,
        perils_covered   JSONB DEFAULT '[]',
        perils_excluded  JSONB DEFAULT '[]',
        effective_date   DATE NOT NULL,
        expiration_date  DATE NOT NULL,
        status           TEXT NOT NULL,    -- active | lapsed | cancelled | pending
        premium_annual   NUMERIC(12,2)
    );

    -- payment ledger
    CREATE TABLE payment_ledger (
        payment_id       TEXT PRIMARY KEY,
        claim_id         TEXT NOT NULL,
        policy_id        TEXT,
        amount           NUMERIC(12,2) NOT NULL,
        payee            TEXT NOT NULL,
        ach_reference    TEXT,
        status           TEXT NOT NULL,    -- authorized | settled | failed | reversed
        idempotency_key  TEXT UNIQUE,      -- for duplicate detection
        created_at       TIMESTAMPTZ DEFAULT now()
    );
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Abstract repository interfaces
# ---------------------------------------------------------------------------
@runtime_checkable
class ClaimsRepository(Protocol):
    """Read/write interface for the claims table."""

    def get_claim(self, claim_id: str) -> Optional[dict[str, Any]]:
        """Return the claim row, or ``None`` if not found."""
        ...

    def update_status(
        self, claim_id: str, new_status: str, *, updated_by: str = "system"
    ) -> Optional[dict[str, Any]]:
        """Update a claim's status. Returns the updated row, or ``None`` if
        the claim doesn't exist."""
        ...


@runtime_checkable
class PolicyRepository(Protocol):
    """Read interface for the policies table."""

    def get_policy(self, policy_id: str) -> Optional[dict[str, Any]]:
        """Return the policy row, or ``None`` if not found."""
        ...


@runtime_checkable
class PaymentLedgerRepository(Protocol):
    """Read/write interface for the payment ledger."""

    def find_by_claim(self, claim_id: str) -> list[dict[str, Any]]:
        """Return all payment records for the given claim."""
        ...

    def find_by_idempotency_key(self, key: str) -> Optional[dict[str, Any]]:
        """Return the payment with the given idempotency key, or ``None``.

        Used for duplicate-payment detection: if a key already exists, the
        caller must NOT initiate a new ACH transfer.
        """
        ...

    def insert(self, record: dict[str, Any]) -> dict[str, Any]:
        """Insert a new payment record. Returns the inserted row."""
        ...


# ---------------------------------------------------------------------------
# In-memory implementations (used by tests and the default registry)
# ---------------------------------------------------------------------------
class InMemoryClaimsRepository:
    """In-memory claims store, seeded with the ShieldPoint demo dataset.

    Schema matches the production PostgreSQL ``claims`` table so a future
    ``PostgresClaimsRepository`` is a drop-in replacement.
    """

    DEFAULT_CLAIMS: list[dict[str, Any]] = [
        {
            "claim_id": "CLM-2026-0001",
            "policy_id": "HO-2024-001",
            "claimant": "Alice Homeowner",
            "amount": 1_250.00,
            "description": "Wind damage to roof shingles during storm on 2026-03-14.",
            "date_of_loss": "2026-03-14",
            "status": "submitted",
            "adjuster_id": "ADJ-42",
            "documents": ["photos_roof_damage.pdf", "contractor_estimate.pdf"],
        },
        {
            "claim_id": "CLM-2026-0002",
            "policy_id": "AU-2024-015",
            "claimant": "Bob Driver",
            "amount": 4_800.00,
            "description": "Collision damage from rear-end accident.",
            "date_of_loss": "2026-04-02",
            "status": "submitted",
            "adjuster_id": "ADJ-43",
            "documents": ["police_report.pdf", "medical_report.pdf"],
        },
        {
            "claim_id": "CLM-2026-0003",
            "policy_id": "HO-2024-088",
            "claimant": "Carol Resident",
            "amount": 250.00,
            "description": "Minor hail damage to mailbox and fence.",
            "date_of_loss": "2026-05-10",
            "status": "validating",
            "adjuster_id": "ADJ-44",
            "documents": ["photos_hail_damage.pdf"],
        },
        {
            "claim_id": "CLM-2026-0004",
            "policy_id": "HO-2024-012",
            "claimant": "Dan Property",
            "amount": 12_500.00,
            "description": "Flood damage to basement. Misrepresentation suspected.",
            "date_of_loss": "2026-02-28",
            "status": "escalating",
            "adjuster_id": "ADJ-45",
            "documents": ["photos_basement.pdf", "hydrology_report.pdf"],
        },
    ]

    def __init__(self, claims: Optional[list[dict[str, Any]]] = None) -> None:
        self._claims: dict[str, dict[str, Any]] = {}
        for c in claims if claims is not None else self.DEFAULT_CLAIMS:
            self._claims[c["claim_id"]] = dict(c)

    def get_claim(self, claim_id: str) -> Optional[dict[str, Any]]:
        c = self._claims.get(claim_id)
        return dict(c) if c else None

    def update_status(
        self, claim_id: str, new_status: str, *, updated_by: str = "system"
    ) -> Optional[dict[str, Any]]:
        c = self._claims.get(claim_id)
        if c is None:
            return None
        c["status"] = new_status
        c["updated_at"] = time.time()
        c["updated_by"] = updated_by
        return dict(c)

    # Test helpers
    def seed(self, claim: dict[str, Any]) -> None:
        self._claims[claim["claim_id"]] = dict(claim)

    def all_claims(self) -> list[dict[str, Any]]:
        return [dict(c) for c in self._claims.values()]


class InMemoryPolicyRepository:
    """In-memory policy store, seeded with the ShieldPoint demo dataset."""

    DEFAULT_POLICIES: list[dict[str, Any]] = [
        {
            "policy_id": "HO-2024-001",
            "type": "homeowners",
            "policyholder": "Alice Homeowner",
            "limit": 250_000,
            "deductible": 1_000,
            "perils_covered": ["wind", "hail", "fire", "theft", "vandalism", "lightning"],
            "perils_excluded": ["flood", "earthquake", "wear_and_tear", "mold"],
            "effective_date": "2024-01-01",
            "expiration_date": "2027-01-01",
            "status": "active",
            "premium_annual": 1_850.00,
        },
        {
            "policy_id": "AU-2024-015",
            "type": "auto",
            "policyholder": "Bob Driver",
            "limit": 50_000,
            "deductible": 500,
            "perils_covered": ["collision", "comprehensive", "uninsured_motorist"],
            "perils_excluded": ["racing", "intentional_damage", "wear_and_tear"],
            "effective_date": "2024-03-15",
            "expiration_date": "2027-03-15",
            "status": "active",
            "premium_annual": 2_400.00,
        },
        {
            "policy_id": "HO-2024-088",
            "type": "homeowners",
            "policyholder": "Carol Resident",
            "limit": 150_000,
            "deductible": 500,
            "perils_covered": ["wind", "hail", "fire", "theft"],
            "perils_excluded": ["flood", "earthquake"],
            "effective_date": "2024-06-01",
            "expiration_date": "2027-06-01",
            "status": "active",
            "premium_annual": 1_200.00,
        },
        {
            "policy_id": "HO-2024-012",
            "type": "homeowners",
            "policyholder": "Dan Property",
            "limit": 300_000,
            "deductible": 2_500,
            "perils_covered": ["wind", "hail", "fire", "theft", "vandalism"],
            "perils_excluded": ["flood", "earthquake", "wear_and_tear", "mold", "intentional_damage"],
            "effective_date": "2024-02-01",
            "expiration_date": "2027-02-01",
            "status": "active",
            "premium_annual": 2_100.00,
        },
        {
            "policy_id": "HO-2023-EXPIRED",
            "type": "homeowners",
            "policyholder": "Eve Lapsed",
            "limit": 100_000,
            "deductible": 1_000,
            "perils_covered": ["wind", "hail", "fire"],
            "perils_excluded": ["flood", "earthquake"],
            "effective_date": "2022-01-01",
            "expiration_date": "2023-01-01",
            "status": "lapsed",
            "premium_annual": 900.00,
        },
    ]

    def __init__(self, policies: Optional[list[dict[str, Any]]] = None) -> None:
        self._policies: dict[str, dict[str, Any]] = {}
        for p in policies if policies is not None else self.DEFAULT_POLICIES:
            self._policies[p["policy_id"]] = dict(p)

    def get_policy(self, policy_id: str) -> Optional[dict[str, Any]]:
        p = self._policies.get(policy_id)
        return dict(p) if p else None

    def seed(self, policy: dict[str, Any]) -> None:
        self._policies[policy["policy_id"]] = dict(policy)

    def all_policies(self) -> list[dict[str, Any]]:
        return [dict(p) for p in self._policies.values()]


class InMemoryPaymentLedgerRepository:
    """In-memory payment ledger with idempotency-key deduplication."""

    def __init__(self, records: Optional[list[dict[str, Any]]] = None) -> None:
        self._records: list[dict[str, Any]] = list(records or [])

    def find_by_claim(self, claim_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._records if r["claim_id"] == claim_id]

    def find_by_idempotency_key(self, key: str) -> Optional[dict[str, Any]]:
        for r in self._records:
            if r.get("idempotency_key") == key:
                return dict(r)
        return None

    def insert(self, record: dict[str, Any]) -> dict[str, Any]:
        # Ensure required fields
        if "payment_id" not in record:
            record["payment_id"] = f"PMT-{uuid.uuid4().hex[:12].upper()}"
        if "idempotency_key" not in record:
            record["idempotency_key"] = record["payment_id"]
        if "created_at" not in record:
            record["created_at"] = time.time()
        record = dict(record)
        self._records.append(record)
        return dict(record)

    def all_records(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._records]
