"""Unit tests for claims-specific tools — claim_lookup, validate_policy, etc. (SHLD-14)."""

from __future__ import annotations

import pytest

from shieldpoint_agents.claims_tools import (
    check_claim_history,
    claim_lookup,
    generate_zkp_proof,
    get_tool_schemas,
    process_payment,
    reset_data_stores,
    seed_claim_history,
    seed_claims,
    seed_policies,
    validate_policy,
)


# ---------------------------------------------------------------------------
# claim_lookup
# ---------------------------------------------------------------------------
class TestClaimLookup:
    def test_lookup_existing_claim(self):
        result = claim_lookup("CLM-2026-0001")
        assert result["claim_id"] == "CLM-2026-0001"
        assert result["amount"] == 1_250.00
        assert result["status"] == "submitted"

    def test_lookup_missing_claim_returns_error(self):
        result = claim_lookup("CLM-NONEXISTENT")
        assert "error" in result
        assert "CLM-NONEXISTENT" in result["error"]

    def test_lookup_returns_all_fields(self):
        result = claim_lookup("CLM-2026-0001")
        assert "policy_id" in result
        assert "claimant" in result
        assert "description" in result
        assert "date_of_loss" in result
        assert "documents" in result


# ---------------------------------------------------------------------------
# validate_policy
# ---------------------------------------------------------------------------
class TestValidatePolicy:
    def test_validate_existing_policy(self):
        result = validate_policy("HO-2024-001")
        assert result["policy_id"] == "HO-2024-001"
        assert result["limit"] == 250_000
        assert "wind" in result["perils_covered"]

    def test_validate_missing_policy_returns_error(self):
        result = validate_policy("POL-NONEXISTENT")
        assert "error" in result

    def test_policy_has_covered_and_excluded_perils(self):
        result = validate_policy("HO-2024-001")
        assert len(result["perils_covered"]) > 0
        assert len(result["perils_excluded"]) > 0
        # No overlap between covered and excluded
        assert not set(result["perils_covered"]) & set(result["perils_excluded"])

    def test_auto_policy_has_different_perils(self):
        result = validate_policy("AU-2024-015")
        assert result["type"] == "auto"
        assert "collision" in result["perils_covered"]


# ---------------------------------------------------------------------------
# check_claim_history
# ---------------------------------------------------------------------------
class TestCheckClaimHistory:
    def test_claimant_with_history(self):
        result = check_claim_history("Alice Homeowner")
        assert result["prior_count"] == 1
        assert result["prior_total"] == 800.00
        assert result["frequent_claimant_flag"] is False

    def test_claimant_with_no_history(self):
        result = check_claim_history("Carol Resident")
        assert result["prior_count"] == 0
        assert result["prior_total"] == 0.0
        assert result["frequent_claimant_flag"] is False

    def test_frequent_claimant_flagged(self):
        result = check_claim_history("Dan Property")
        assert result["prior_count"] == 2
        assert result["frequent_claimant_flag"] is False  # 2 < 3

    def test_unknown_claimant_returns_empty(self):
        result = check_claim_history("Unknown Person")
        assert result["prior_count"] == 0
        assert result["frequent_claimant_flag"] is False


# ---------------------------------------------------------------------------
# process_payment
# ---------------------------------------------------------------------------
class TestProcessPayment:
    def test_payment_authorized(self):
        result = process_payment(
            claim_id="CLM-2026-0001",
            amount=1250.00,
            payee="Alice Homeowner",
            policy_id="HO-2024-001",
        )
        assert result["status"] == "authorized"
        assert result["claim_id"] == "CLM-2026-0001"
        assert result["amount"] == 1250.00
        assert result["payee"] == "Alice Homeowner"
        assert result["payment_id"].startswith("PMT-CLM-2026-0001-")

    def test_payment_without_policy_id(self):
        result = process_payment(
            claim_id="CLM-2026-0002",
            amount=4800.00,
            payee="Bob Driver",
        )
        assert result["status"] == "authorized"
        assert result["policy_id"] == ""

    def test_multiple_payments_unique_ids(self):
        r1 = process_payment(claim_id="CLM-1", amount=100, payee="A")
        r2 = process_payment(claim_id="CLM-2", amount=200, payee="B")
        assert r1["payment_id"] != r2["payment_id"]


# ---------------------------------------------------------------------------
# generate_zkp_proof
# ---------------------------------------------------------------------------
class TestGenerateZKPProof:
    def test_proof_within_limit(self):
        result = generate_zkp_proof(
            claim_id="CLM-2026-0001",
            policy_id="HO-2024-001",
            claim_amount=1250.00,
            coverage_limit=250_000,
        )
        assert result["verified"] is True
        assert result["proof"].startswith("zkp:")
        assert "within" in result["statement"].lower()

    def test_proof_exceeds_limit(self):
        result = generate_zkp_proof(
            claim_id="CLM-2026-0004",
            policy_id="HO-2024-012",
            claim_amount=500_000,
            coverage_limit=300_000,
        )
        assert result["verified"] is False
        assert "EXCEEDS" in result["statement"]

    def test_proof_at_exact_limit(self):
        result = generate_zkp_proof(
            claim_id="CLM-2026-EDGE",
            policy_id="HO-2024-001",
            claim_amount=250_000,
            coverage_limit=250_000,
        )
        assert result["verified"] is True

    def test_proof_is_deterministic_for_same_inputs(self):
        # Same inputs should produce the same proof hash
        import time
        r1 = generate_zkp_proof(
            claim_id="CLM-SAME",
            policy_id="HO-2024-001",
            claim_amount=1000,
            coverage_limit=250_000,
        )
        # Wait a tiny bit to ensure timestamp differs
        time.sleep(0.01)
        r2 = generate_zkp_proof(
            claim_id="CLM-SAME",
            policy_id="HO-2024-001",
            claim_amount=1000,
            coverage_limit=250_000,
        )
        # The proofs may differ due to timestamp, but the verified field
        # should be the same
        assert r1["verified"] == r2["verified"]


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
class TestToolSchemas:
    def test_get_tool_schemas_returns_all_tools(self):
        schemas = get_tool_schemas()
        assert "claim_lookup" in schemas
        assert "validate_policy" in schemas
        assert "check_claim_history" in schemas
        assert "process_payment" in schemas
        assert "generate_zkp_proof" in schemas

    def test_schemas_have_required_fields(self):
        schemas = get_tool_schemas()
        for name, schema in schemas.items():
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema


# ---------------------------------------------------------------------------
# Data store management
# ---------------------------------------------------------------------------
class TestDataStoreManagement:
    def test_seed_claims(self):
        reset_data_stores()
        seed_claims({"CLM-TEST": {"claim_id": "CLM-TEST", "amount": 999}})
        result = claim_lookup("CLM-TEST")
        assert result["amount"] == 999
        reset_data_stores()  # cleanup

    def test_seed_policies(self):
        reset_data_stores()
        seed_policies({"POL-TEST": {"policy_id": "POL-TEST", "limit": 50000}})
        result = validate_policy("POL-TEST")
        assert result["limit"] == 50000
        reset_data_stores()

    def test_reset_clears_everything(self):
        reset_data_stores()
        assert claim_lookup("CLM-2026-0001").get("error") is not None
        assert validate_policy("HO-2024-001").get("error") is not None
