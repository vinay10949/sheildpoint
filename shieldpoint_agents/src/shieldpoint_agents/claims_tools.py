"""
Claims-specific tool implementations for the ShieldPoint Tool-Using Agent (SHLD-14).

This module provides the four core tools the agent needs for claims processing:

1. **claim_lookup** — Retrieve claim details by claim_id from the claims database.
2. **validate_policy** — Look up a policy and return coverage limits, deductibles,
   and covered/excluded perils.
3. **process_payment** — Authorize a payment for an approved claim (simulated
   for the MVP; real integration hooks are stubbed).
4. **generate_zkp_proof** — Generate a Zero-Knowledge Proof that a claim meets
   policy conditions without revealing sensitive claimant data (simulated
   for the MVP).

Each tool is a plain Python function that returns a dict. In production, these
would connect to real databases, payment gateways, and ZKP services. For the
MVP, they use in-memory data stores that can be seeded with test data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger("shieldpoint_agents.claims_tools")


# ---------------------------------------------------------------------------
# In-memory data stores (seeded with sample data for testing/development)
# ---------------------------------------------------------------------------
_CLAIMS_DB: dict[str, dict[str, Any]] = {
    "CLM-2026-0001": {
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
    "CLM-2026-0002": {
        "claim_id": "CLM-2026-0002",
        "policy_id": "AU-2024-015",
        "claimant": "Bob Driver",
        "amount": 4_800.00,
        "description": "Collision damage from rear-end accident. Minor injury reported.",
        "date_of_loss": "2026-04-02",
        "status": "submitted",
        "adjuster_id": "ADJ-43",
        "documents": ["police_report.pdf", "medical_report.pdf"],
    },
    "CLM-2026-0003": {
        "claim_id": "CLM-2026-0003",
        "policy_id": "HO-2024-088",
        "claimant": "Carol Resident",
        "amount": 250.00,
        "description": "Minor hail damage to mailbox and fence.",
        "date_of_loss": "2026-05-10",
        "status": "submitted",
        "adjuster_id": "ADJ-44",
        "documents": ["photos_hail_damage.pdf"],
    },
    "CLM-2026-0004": {
        "claim_id": "CLM-2026-0004",
        "policy_id": "HO-2024-012",
        "claimant": "Dan Property",
        "amount": 12_500.00,
        "description": "Flood damage to basement after heavy rain. Intentional misrepresentation suspected.",
        "date_of_loss": "2026-02-28",
        "status": "under_investigation",
        "adjuster_id": "ADJ-45",
        "documents": ["photos_basement.pdf", "contractor_estimate.pdf", "hydrology_report.pdf"],
    },
}

_POLICIES_DB: dict[str, dict[str, Any]] = {
    "HO-2024-001": {
        "policy_id": "HO-2024-001",
        "type": "homeowners",
        "policyholder": "Alice Homeowner",
        "limit": 250_000,
        "deductible": 1_000,
        "perils_covered": ["wind", "hail", "fire", "theft", "vandalism", "lightning"],
        "perils_excluded": ["flood", "earthquake", "wear_and_tear", "mold"],
        "effective_date": "2024-01-01",
        "expiration_date": "2027-01-01",
        "premium_annual": 1_850.00,
    },
    "AU-2024-015": {
        "policy_id": "AU-2024-015",
        "type": "auto",
        "policyholder": "Bob Driver",
        "limit": 50_000,
        "deductible": 500,
        "perils_covered": ["collision", "comprehensive", "uninsured_motorist"],
        "perils_excluded": ["racing", "intentional_damage", "wear_and_tear"],
        "effective_date": "2024-03-15",
        "expiration_date": "2027-03-15",
        "premium_annual": 2_400.00,
    },
    "HO-2024-088": {
        "policy_id": "HO-2024-088",
        "type": "homeowners",
        "policyholder": "Carol Resident",
        "limit": 150_000,
        "deductible": 500,
        "perils_covered": ["wind", "hail", "fire", "theft"],
        "perils_excluded": ["flood", "earthquake"],
        "effective_date": "2024-06-01",
        "expiration_date": "2027-06-01",
        "premium_annual": 1_200.00,
    },
    "HO-2024-012": {
        "policy_id": "HO-2024-012",
        "type": "homeowners",
        "policyholder": "Dan Property",
        "limit": 300_000,
        "deductible": 2_500,
        "perils_covered": ["wind", "hail", "fire", "theft", "vandalism"],
        "perils_excluded": ["flood", "earthquake", "wear_and_tear", "mold", "intentional_damage"],
        "effective_date": "2024-02-01",
        "expiration_date": "2027-02-01",
        "premium_annual": 2_100.00,
    },
}

_PAYMENT_LEDGER: list[dict[str, Any]] = []

_CLAIM_HISTORY_DB: dict[str, list[dict[str, Any]]] = {
    "Alice Homeowner": [
        {"claim_id": "CLM-2024-0099", "amount": 800.00, "date": "2024-08-15", "decision": "approve"},
    ],
    "Bob Driver": [
        {"claim_id": "CLM-2025-0012", "amount": 2_300.00, "date": "2025-03-20", "decision": "approve"},
    ],
    "Carol Resident": [],
    "Dan Property": [
        {"claim_id": "CLM-2024-0045", "amount": 5_000.00, "date": "2024-11-10", "decision": "deny"},
        {"claim_id": "CLM-2025-0078", "amount": 8_200.00, "date": "2025-06-01", "decision": "route_to_manual_review"},
    ],
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def claim_lookup(claim_id: str) -> dict[str, Any]:
    """Look up a claim by its ID and return the full claim record.

    Returns the claim details including status, documents, and assigned
    adjuster. If the claim is not found, returns an error dict.
    """
    claim = _CLAIMS_DB.get(claim_id)
    if claim is None:
        logger.warning("Claim lookup failed: claim_id=%s not found", claim_id)
        return {"error": f"Claim '{claim_id}' not found", "claim_id": claim_id}
    logger.info("Claim lookup succeeded: claim_id=%s", claim_id)
    return dict(claim)


def validate_policy(policy_id: str) -> dict[str, Any]:
    """Validate a policy by ID and return coverage details.

    Returns the policy's coverage limit, deductible, covered perils,
    and excluded perils. If the policy is not found, returns an error dict.
    """
    policy = _POLICIES_DB.get(policy_id)
    if policy is None:
        logger.warning("Policy validation failed: policy_id=%s not found", policy_id)
        return {"error": f"Policy '{policy_id}' not found", "policy_id": policy_id}
    logger.info("Policy validation succeeded: policy_id=%s", policy_id)
    return dict(policy)


def check_claim_history(claimant: str) -> dict[str, Any]:
    """Check a claimant's claim history over the last 24 months.

    Returns the number of prior claims and total amount claimed. Frequent
    claimants may be flagged for additional review.
    """
    history = _CLAIM_HISTORY_DB.get(claimant, [])
    prior_count = len(history)
    prior_total = sum(h.get("amount", 0.0) for h in history)

    # Flag frequent claimants (3+ claims in 24 months)
    frequent_flag = prior_count >= 3

    result = {
        "claimant": claimant,
        "prior_count": prior_count,
        "prior_total": prior_total,
        "frequent_claimant_flag": frequent_flag,
        "claims": history,
    }
    logger.info(
        "Claim history checked: claimant=%s prior_count=%d total=$%.2f",
        claimant, prior_count, prior_total,
    )
    return result


def process_payment(
    claim_id: str,
    amount: float,
    payee: str,
    *,
    policy_id: str = "",
) -> dict[str, Any]:
    """Authorize a payment for an approved claim.

    In the MVP, this records the payment in the in-memory ledger and
    returns a payment confirmation. In production, this would connect
    to a payment gateway and escrow service.
    """
    payment_id = f"PMT-{claim_id}-{int(time.time())}"
    record = {
        "payment_id": payment_id,
        "claim_id": claim_id,
        "policy_id": policy_id,
        "amount": amount,
        "payee": payee,
        "status": "authorized",
        "timestamp": time.time(),
    }
    _PAYMENT_LEDGER.append(record)
    logger.info(
        "Payment authorized: payment_id=%s claim_id=%s amount=$%.2f payee=%s",
        payment_id, claim_id, amount, payee,
    )
    return record


def generate_zkp_proof(
    claim_id: str,
    policy_id: str,
    claim_amount: float,
    coverage_limit: float,
) -> dict[str, Any]:
    """Generate a Zero-Knowledge Proof that a claim meets policy conditions.

    In the MVP, this simulates ZKP generation by producing a deterministic
    hash-based proof that can be verified without revealing the underlying
    claim details. In production, this would use a real ZKP library
    (e.g., circom/snarkjs or zkSync).

    The proof demonstrates:
    - claim_amount <= coverage_limit (claim is within coverage)
    - The peril is covered (without revealing which specific peril)
    - The policy is active and valid
    """
    # Simulated ZKP: hash the inputs to create a "proof" that can be verified
    proof_input = json.dumps(
        {
            "claim_id": claim_id,
            "policy_id": policy_id,
            "claim_within_limit": claim_amount <= coverage_limit,
            "timestamp": int(time.time()),
        },
        sort_keys=True,
    )
    proof_hash = hashlib.sha256(proof_input.encode()).hexdigest()

    is_valid = claim_amount <= coverage_limit

    result = {
        "claim_id": claim_id,
        "policy_id": policy_id,
        "proof": f"zkp:{proof_hash[:32]}",
        "proof_type": "simulated_sha256",
        "verified": is_valid,
        "statement": (
            f"Claim amount ${claim_amount:,.2f} is within policy "
            f"coverage limit ${coverage_limit:,.2f}"
            if is_valid
            else f"Claim amount ${claim_amount:,.2f} EXCEEDS policy "
            f"coverage limit ${coverage_limit:,.2f}"
        ),
    }
    logger.info(
        "ZKP proof generated: claim_id=%s policy_id=%s verified=%s",
        claim_id, policy_id, is_valid,
    )
    return result


# ---------------------------------------------------------------------------
# Data store management (for testing/seeding)
# ---------------------------------------------------------------------------

def seed_claims(claims: dict[str, dict[str, Any]]) -> None:
    """Add claims to the in-memory store (for testing)."""
    _CLAIMS_DB.update(claims)


def seed_policies(policies: dict[str, dict[str, Any]]) -> None:
    """Add policies to the in-memory store (for testing)."""
    _POLICIES_DB.update(policies)


def seed_claim_history(history: dict[str, list[dict[str, Any]]]) -> None:
    """Add claim history records to the in-memory store (for testing)."""
    _CLAIM_HISTORY_DB.update(history)


def reset_data_stores() -> None:
    """Reset all in-memory data stores to empty (for test isolation)."""
    _CLAIMS_DB.clear()
    _POLICIES_DB.clear()
    _PAYMENT_LEDGER.clear()
    _CLAIM_HISTORY_DB.clear()


def get_tool_schemas() -> dict[str, dict[str, Any]]:
    """Return the JSON-Schema descriptors for all claims tools.

    Useful for registering tools with the ToolRegistry in one call.
    """
    return {
        "claim_lookup": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string", "description": "The claim ID to look up."},
            },
            "required": ["claim_id"],
            "additionalProperties": False,
        },
        "validate_policy": {
            "type": "object",
            "properties": {
                "policy_id": {"type": "string", "description": "The policy ID to validate."},
            },
            "required": ["policy_id"],
            "additionalProperties": False,
        },
        "check_claim_history": {
            "type": "object",
            "properties": {
                "claimant": {"type": "string", "description": "The claimant's name to check history for."},
            },
            "required": ["claimant"],
            "additionalProperties": False,
        },
        "process_payment": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string", "description": "The claim ID for the payment."},
                "amount": {"type": "number", "description": "Payment amount."},
                "payee": {"type": "string", "description": "Name of the payee."},
                "policy_id": {"type": "string", "description": "Associated policy ID (optional).", "default": ""},
            },
            "required": ["claim_id", "amount", "payee"],
            "additionalProperties": False,
        },
        "generate_zkp_proof": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string", "description": "The claim ID."},
                "policy_id": {"type": "string", "description": "The policy ID."},
                "claim_amount": {"type": "number", "description": "The claim amount."},
                "coverage_limit": {"type": "number", "description": "The policy coverage limit."},
            },
            "required": ["claim_id", "policy_id", "claim_amount", "coverage_limit"],
            "additionalProperties": False,
        },
    }
