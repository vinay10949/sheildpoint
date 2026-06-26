"""Unit tests for the field validator."""

from __future__ import annotations

import pytest

from claim_intake.config import IntakeConfig
from claim_intake.schemas import ClaimType, FieldError
from claim_intake.validator import validate, to_standard_claim


# ---------------------------------------------------------------------------
# Required field validation
# ---------------------------------------------------------------------------
class TestRequiredFields:
    def test_all_valid_fields_pass(self):
        result = validate({
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof shingles during storm.",
        })
        assert result.ok
        assert len(result.errors) == 0
        assert result.cleaned["policyholder_name"] == "Alice Homeowner"
        assert result.cleaned["policy_id"] == "HO-2024-001"
        assert result.cleaned["claim_type"] == ClaimType.HOMEOWNERS

    def test_missing_required_field_reported(self):
        result = validate({
            "policyholder_name": "Alice Homeowner",
            # policy_id missing
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof.",
        })
        assert not result.ok
        assert "policy_id" in result.missing
        assert any(e.field == "policy_id" for e in result.errors)

    def test_all_required_missing(self):
        result = validate({})
        assert not result.ok
        assert len(result.missing) == 5
        assert len(result.errors) == 5

    def test_invalid_policy_id_format(self):
        result = validate({
            "policyholder_name": "Alice Homeowner",
            "policy_id": "not-a-policy-id",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof.",
        })
        assert not result.ok
        assert any(
            e.field == "policy_id" and "format" in e.message.lower()
            for e in result.errors
        )

    def test_invalid_claim_type(self):
        result = validate({
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "watercraft",  # not in enum
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof.",
        })
        assert not result.ok
        assert any(e.field == "claim_type" for e in result.errors)

    def test_invalid_date_format(self):
        result = validate({
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "March 14, 2026",  # not ISO
            "damage_description": "Wind damage to roof.",
        })
        assert not result.ok
        assert any(
            e.field == "date_of_loss" and "iso" in e.message.lower()
            for e in result.errors
        )

    def test_future_date_rejected(self):
        from datetime import date, timedelta
        future = (date.today() + timedelta(days=30)).isoformat()
        result = validate({
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": future,
            "damage_description": "Wind damage to roof.",
        })
        assert not result.ok
        assert any(
            e.field == "date_of_loss" and "future" in e.message.lower()
            for e in result.errors
        )

    def test_old_date_warned(self):
        # >5 years ago — flagged for OCR verification.
        result = validate({
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2010-01-01",
            "damage_description": "Wind damage to roof.",
        })
        assert not result.ok
        assert any(
            e.field == "date_of_loss" and "5 years" in e.message
            for e in result.errors
        )

    def test_short_damage_description_rejected(self):
        result = validate({
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind",  # 4 chars — too short
        })
        assert not result.ok
        assert any(
            e.field == "damage_description" and "short" in e.message.lower()
            for e in result.errors
        )

    def test_ocr_garbage_description_rejected(self):
        # A single 200+ char word with no spaces — classic OCR garbage.
        garbage = "x" * 250
        result = validate({
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": garbage,
        })
        assert not result.ok
        assert any(
            e.field == "damage_description" and "garbage" in e.message.lower()
            for e in result.errors
        )


# ---------------------------------------------------------------------------
# Policyholder name validation
# ---------------------------------------------------------------------------
class TestPolicyholderName:
    def test_single_token_rejected(self):
        result = validate({
            "policyholder_name": "Alice",  # no last name
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof shingles.",
        })
        assert not result.ok
        assert any(e.field == "policyholder_name" for e in result.errors)

    def test_all_caps_title_cased(self):
        result = validate({
            "policyholder_name": "ALICE HOMEOWNER",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof shingles.",
        })
        assert result.ok
        assert result.cleaned["policyholder_name"] == "Alice Homeowner"

    def test_three_part_name_accepted(self):
        result = validate({
            "policyholder_name": "Alice Marie Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof shingles.",
        })
        assert result.ok
        assert result.cleaned["policyholder_name"] == "Alice Marie Homeowner"


# ---------------------------------------------------------------------------
# Optional field validation
# ---------------------------------------------------------------------------
class TestOptionalFields:
    def test_absent_optional_fields_are_not_errors(self):
        result = validate({
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof shingles.",
        })
        assert result.ok
        assert len(result.invalid_optional) == 0

    def test_invalid_amount_rejected(self):
        result = validate({
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof shingles.",
            "amount_claimed": "not-a-number",
        })
        assert not result.ok
        assert "amount_claimed" in result.invalid_optional

    def test_negative_amount_rejected(self):
        result = validate({
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof shingles.",
            "amount_claimed": -500.00,
        })
        assert not result.ok
        assert "amount_claimed" in result.invalid_optional

    def test_short_phone_rejected(self):
        result = validate({
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof shingles.",
            "phone": "555",  # too short
        })
        assert not result.ok
        assert "phone" in result.invalid_optional

    def test_valid_phone_accepted(self):
        result = validate({
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof shingles.",
            "phone": "(555) 123-4567",
        })
        assert result.ok
        assert result.cleaned["phone"] == "(555) 123-4567"

    def test_invalid_email_rejected(self):
        result = validate({
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof shingles.",
            "email": "not-an-email",
        })
        assert not result.ok
        assert "email" in result.invalid_optional


# ---------------------------------------------------------------------------
# to_standard_claim
# ---------------------------------------------------------------------------
class TestToStandardClaim:
    def test_builds_claim_from_cleaned(self):
        cleaned = {
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": ClaimType.HOMEOWNERS,
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof.",
        }
        claim = to_standard_claim(cleaned)
        assert claim.policyholder_name == "Alice Homeowner"
        assert claim.policy_id == "HO-2024-001"
        assert claim.claim_type == ClaimType.HOMEOWNERS

    def test_strips_unknown_fields(self):
        cleaned = {
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": ClaimType.HOMEOWNERS,
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof.",
            "unknown_field": "should be ignored",
        }
        # Should not raise — to_standard_claim filters by model_fields.
        claim = to_standard_claim(cleaned)
        assert claim.policyholder_name == "Alice Homeowner"

    def test_raises_on_missing_required(self):
        with pytest.raises(Exception):
            to_standard_claim({"policyholder_name": "Alice"})
