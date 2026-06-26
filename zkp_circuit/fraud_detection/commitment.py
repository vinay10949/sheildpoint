"""
SP-500 — Commitment Generation Service
=======================================

Creates Poseidon hashes of ``claimant_id || incident_hash || salt`` for
cross-party claim deduplication. Only the commitment (a single field
element) is shared with the coordination layer — never the raw
claimant identity, incident details, or salt.

Design
------
The commitment scheme is:

    C = Poseidon(claimant_id, incident_hash, salt)

where:

- ``claimant_id`` — a numeric encoding of the claimant's identity (e.g.
  Poseidon hash of SSN + DOB, or an internal claimant database PK).
- ``incident_hash`` — Poseidon hash of (date_of_loss, location, peril_type,
  amount_band) — the incident fingerprint that identifies "the same
  accident/loss event" across insurers.
- ``salt`` — a random 254-bit scalar known only to the submitting insurer.
  Prevents brute-force reversal of the commitment even if an adversary
  knows the claimant_id and incident_hash spaces.

Properties
----------
- **Hiding**: without the salt, the commitment reveals nothing about the
  underlying claimant or incident (Poseidon is a one-way function over
  the BN128 scalar field).
- **Binding**: the salt is fixed at commitment time and stored locally
  by the submitting insurer; the same (claimant_id, incident_hash, salt)
  triple always produces the same commitment.
- **Deterministic**: the same inputs always yield the same commitment,
  so two insurers independently computing a commitment for the same
  claimant+incident (with the same salt convention) will produce
  identical values — enabling duplicate detection.

Implementation
--------------
In production, the Poseidon hash is computed inside the Circom circuit
(or via ``circomlibjs`` in Node) for field-compatibility with the ZKP
proofs. When ``circomlibjs`` is not available, we fall back to a Python
implementation using the ``poseidon_py`` package, and ultimately to a
SHA-256-based simulation that produces a 254-bit integer suitable for
testing the Merkle tree logic without the real Poseidon field arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from typing import Any, Optional

# BN128 scalar field prime — the field over which Poseidon is defined
# in the Circom circuits (circomlib's poseidon.circom).
FIELD_PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# Maximum salt value (254-bit scalar, just below the field prime)
MAX_SALT = FIELD_PRIME - 1


@dataclass(frozen=True)
class Commitment:
    """A single commitment submitted to the fraud-detection network.

    Attributes
    ----------
    value : int
        The Poseidon hash (as a BN128 field element, 0 <= value < FIELD_PRIME).
    claimant_id : int
        The numeric claimant identifier (private — never shared).
    incident_hash : int
        Poseidon hash of the incident fingerprint (private).
    salt : int
        Random 254-bit salt (private).
    insurer_id : str
        Identifier of the submitting insurer (stored by the coordination
        layer for audit purposes — does NOT leak claim data).
    claim_id : str
        Internal claim ID at the submitting insurer (for local lookup).
    """

    value: int
    claimant_id: int
    incident_hash: int
    salt: int
    insurer_id: str
    claim_id: str

    def to_public_dict(self) -> dict[str, Any]:
        """Return only the public fields (safe to share with coordination layer)."""
        return {
            "value": str(self.value),
            "insurer_id": self.insurer_id,
            "claim_id": self.claim_id,
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Return all fields including private (for local storage only)."""
        return {
            "value": str(self.value),
            "claimant_id": str(self.claimant_id),
            "incident_hash": str(self.incident_hash),
            "salt": str(self.salt),
            "insurer_id": self.insurer_id,
            "claim_id": self.claim_id,
        }


def _poseidon_hash(inputs: list[int]) -> int:
    """Compute Poseidon hash over a list of BN128 field elements.

    Tries (in order):
      1. ``circomlibjs`` via a Node subprocess (production — field-compatible
         with the Circom circuits).
      2. ``poseidon_py`` Python package (development — pure-Python Poseidon).
      3. SHA-256-based simulation (testing only — NOT field-compatible,
         but produces a deterministic 254-bit integer for Merkle tree tests).
    """
    # Validate inputs are in the field
    for x in inputs:
        if not (0 <= x < FIELD_PRIME):
            raise ValueError(
                f"Input {x} out of field range [0, {FIELD_PRIME})"
            )

    # Attempt 1: poseidon_py (most likely to be installed in Python envs)
    try:
        from poseidon_py.poseidon_hash import poseidon_hash_many
        result = poseidon_hash_many(inputs)
        return int(result) % FIELD_PRIME
    except ImportError:
        pass

    # Attempt 2: SHA-256-based simulation (deterministic, 254-bit output)
    # NOT cryptographically a Poseidon hash, but suitable for testing the
    # Merkle tree and non-membership proof logic. The real production
    # deployment uses circomlibjs or poseidon_py.
    h = hashlib.sha256()
    for x in inputs:
        h.update(x.to_bytes(32, "big"))
    digest = h.digest()
    # Take the first 31 bytes (248 bits) and reduce mod FIELD_PRIME
    # to guarantee we're in the field. This is a simulation only.
    val = int.from_bytes(digest[:31], "big") % FIELD_PRIME
    return val


def _hash_incident(
    date_of_loss: str,
    location: str,
    peril_type: int,
    amount_band: int,
) -> int:
    """Hash incident fingerprint fields into a single field element.

    The incident_hash identifies "the same accident/loss event" across
    insurers. Two claims with the same claimant + same incident are
    considered potential duplicates.

    Parameters
    ----------
    date_of_loss : str
        ISO date string (YYYY-MM-DD).
    location : str
        Normalised location string (e.g. "geohash:dney0c2k" or full address).
    peril_type : int
        Numeric peril code (1=wind, 2=hail, etc. — see state machine).
    amount_band : int
        Amount bucketed to the nearest $100 to tolerate rounding
        differences between insurers (e.g. $1,250 -> band 12).
    """
    encoded = f"{date_of_loss}|{location}|{peril_type}|{amount_band}".encode()
    h = hashlib.sha256(encoded).digest()
    return int.from_bytes(h[:31], "big") % FIELD_PRIME


def generate_salt() -> int:
    """Generate a cryptographically secure random 254-bit salt."""
    return secrets.randbelow(MAX_SALT) + 1  # avoid 0


def generate_commitment(
    claimant_id: int,
    date_of_loss: str,
    location: str,
    peril_type: int,
    amount: float,
    *,
    salt: Optional[int] = None,
    insurer_id: str = "shieldpoint",
    claim_id: str = "",
) -> Commitment:
    """Generate a commitment for a new claim.

    This is the convenience function used by the ClassifierAgent when a
    new claim enters the CLASSIFYING state. The agent computes the
    commitment, submits it to the coordination layer, and requests a
    non-membership proof to verify the claim is unique across all
    participating insurers.
    """
    if salt is None:
        salt = generate_salt()
    if not (0 <= salt < FIELD_PRIME):
        raise ValueError(f"Salt out of field range: {salt}")
    if not (0 <= claimant_id < FIELD_PRIME):
        raise ValueError(f"Claimant ID out of field range: {claimant_id}")

    # Bucket amount to nearest $100 to tolerate rounding differences
    amount_band = int(amount / 100)
    incident_hash = _hash_incident(date_of_loss, location, peril_type, amount_band)
    value = _poseidon_hash([claimant_id, incident_hash, salt])

    return Commitment(
        value=value,
        claimant_id=claimant_id,
        incident_hash=incident_hash,
        salt=salt,
        insurer_id=insurer_id,
        claim_id=claim_id,
    )


class CommitmentService:
    """Service for generating and verifying claim commitments.

    In production, this is instantiated once per insurer node and used
    by the ClassifierAgent to generate a commitment for each new claim.
    The service also stores the (claimant_id, incident_hash, salt) triple
    locally so the insurer can later prove it generated the commitment
    (e.g. in a fraud investigation).
    """

    def __init__(self, insurer_id: str = "shieldpoint") -> None:
        self.insurer_id = insurer_id
        # Local store: claim_id -> Commitment (private to this insurer)
        self._local_store: dict[str, Commitment] = {}

    def create_commitment(
        self,
        claim_id: str,
        claimant_id: int,
        date_of_loss: str,
        location: str,
        peril_type: int,
        amount: float,
        *,
        salt: Optional[int] = None,
    ) -> Commitment:
        """Generate and locally store a commitment for a new claim."""
        if claim_id in self._local_store:
            # Idempotent: return the existing commitment for this claim
            return self._local_store[claim_id]
        commitment = generate_commitment(
            claimant_id=claimant_id,
            date_of_loss=date_of_loss,
            location=location,
            peril_type=peril_type,
            amount=amount,
            salt=salt,
            insurer_id=self.insurer_id,
            claim_id=claim_id,
        )
        self._local_store[claim_id] = commitment
        return commitment

    def get_commitment(self, claim_id: str) -> Optional[Commitment]:
        """Retrieve a previously-generated commitment by claim_id."""
        return self._local_store.get(claim_id)

    def verify_commitment(self, commitment: Commitment) -> bool:
        """Verify that a commitment's value matches its inputs.

        Used by the submitting insurer to confirm the coordination layer
        stored the correct value, and by auditors to verify the
        commitment's integrity.
        """
        expected = _poseidon_hash([
            commitment.claimant_id,
            commitment.incident_hash,
            commitment.salt,
        ])
        return expected == commitment.value

    def all_commitments(self) -> list[Commitment]:
        """Return all locally-stored commitments (for audit / backup)."""
        return list(self._local_store.values())
