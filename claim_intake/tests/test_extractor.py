"""Unit tests for the field extractor — regex + LLM-assisted parsing."""

from __future__ import annotations

import pytest

from claim_intake.config import IntakeConfig
from claim_intake.extractor import (
    extract,
    _normalise_claim_type,
    _normalise_date,
)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
class TestNormaliseDate:
    def test_iso_passthrough(self):
        assert _normalise_date("2026-03-14") == "2026-03-14"

    def test_us_to_iso(self):
        assert _normalise_date("03/14/2026") == "2026-03-14"

    def test_us_short_year(self):
        assert _normalise_date("3/14/26") == "2026-03-14"

    def test_invalid_passthrough(self):
        # Invalid dates pass through unchanged — the validator catches them.
        assert _normalise_date("not-a-date") == "not-a-date"


class TestNormaliseClaimType:
    def test_known_lowercase(self):
        from claim_intake.schemas import ClaimType
        assert _normalise_claim_type("homeowners") == ClaimType.HOMEOWNERS
        assert _normalise_claim_type("auto") == ClaimType.AUTO

    def test_known_uppercase(self):
        from claim_intake.schemas import ClaimType
        assert _normalise_claim_type("HOMEOWNERS") == ClaimType.HOMEOWNERS

    def test_plural(self):
        from claim_intake.schemas import ClaimType
        assert _normalise_claim_type("homeowner") == ClaimType.HOMEOWNERS

    def test_unknown_falls_back_to_other(self):
        from claim_intake.schemas import ClaimType
        assert _normalise_claim_type("watercraft") == ClaimType.OTHER

    def test_keyword_inference(self):
        from claim_intake.schemas import ClaimType
        assert _normalise_claim_type("this is about my car") == ClaimType.AUTO
        assert _normalise_claim_type("roof leak at home") == ClaimType.HOMEOWNERS


# ---------------------------------------------------------------------------
# Regex extraction
# ---------------------------------------------------------------------------
class TestRegexExtraction:
    def test_extracts_all_labelled_fields(self):
        text = """
            Policyholder Name: Alice Homeowner
            Policy ID: HO-2024-001
            Claim Type: homeowners
            Date of Loss: 2026-03-14
            Amount Claimed: $1,250.00
            Phone: (555) 123-4567
            Email: alice@example.com
            Location: 123 Main St, Anytown, USA

            Damage Description:
            Wind damage to roof shingles during severe thunderstorm.
            Approximately 30% of shingles blown off.
        """
        result = extract(text, config=IntakeConfig(llm_base_url=""))
        assert result.fields["policyholder_name"] == "Alice Homeowner"
        assert result.fields["policy_id"] == "HO-2024-001"
        assert result.fields["claim_type"] == "homeowners"
        assert result.fields["date_of_loss"] == "2026-03-14"
        assert result.fields["amount_claimed"] == 1250.00
        assert result.fields["phone"] == "(555) 123-4567"
        assert result.fields["email"] == "alice@example.com"
        assert result.fields.get("incident_location") is not None
        assert "Wind damage" in result.fields["damage_description"]
        # All target fields populated.
        assert result.missing_count == 0

    def test_extracts_bare_policy_id_when_no_label(self):
        text = """
            Claim form received from Alice.
            HO-2024-001
            Date of Loss: 2026-03-14
            Description: Wind damage to roof.
        """
        result = extract(text, config=IntakeConfig(llm_base_url=""))
        assert result.fields["policy_id"] == "HO-2024-001"

    def test_us_date_normalised_to_iso(self):
        text = "Date of Loss: 03/14/2026\nDescription: Storm damage."
        result = extract(text, config=IntakeConfig(llm_base_url=""))
        assert result.fields["date_of_loss"] == "2026-03-14"

    def test_claim_type_inferred_from_description_when_label_missing(self):
        text = """
            Policyholder Name: Bob Driver
            Policy ID: AU-2024-015
            Date of Loss: 2026-04-02

            Damage Description:
            Rear-end collision on highway. Front bumper crushed.
        """
        result = extract(text, config=IntakeConfig(llm_base_url=""))
        # No explicit claim_type label — should infer "auto" from "collision".
        assert result.fields.get("claim_type") == "auto"

    def test_description_fallback_to_longest_paragraph(self):
        text = """
            Policyholder Name: Carol Resident
            Policy ID: HO-2024-088

            This is a short intro paragraph that should not be picked.

            This is a much longer paragraph that should be picked as the
            damage description because it is the longest block of text
            in the entire document. Hail damaged the mailbox and fence.
        """
        result = extract(text, config=IntakeConfig(llm_base_url=""))
        desc = result.fields.get("damage_description")
        assert desc is not None
        assert "longer paragraph" in desc

    def test_missing_fields_counted(self):
        text = "Just some random text with no claim fields."
        result = extract(text, config=IntakeConfig(llm_base_url=""))
        assert result.missing_count >= 4  # most required fields missing
        assert "policyholder_name" not in result.fields
        assert "policy_id" not in result.fields

    def test_amount_with_commas_parsed(self):
        text = "Amount: $12,500.00\nDescription: Flood damage."
        result = extract(text, config=IntakeConfig(llm_base_url=""))
        assert result.fields["amount_claimed"] == 12500.00


# ---------------------------------------------------------------------------
# Passthrough values
# ---------------------------------------------------------------------------
class TestPassthrough:
    def test_passthrough_takes_precedence(self):
        text = "Policyholder Name: OCR Wrong Name\nPolicy ID: HO-2024-001"
        result = extract(
            text,
            config=IntakeConfig(llm_base_url=""),
            passthrough={"policyholder_name": "Alice Homeowner"},
        )
        assert result.fields["policyholder_name"] == "Alice Homeowner"
        assert result.method["policyholder_name"] == "passthrough"

    def test_empty_passthrough_ignored(self):
        text = "Policyholder Name: Alice Homeowner"
        result = extract(
            text,
            config=IntakeConfig(llm_base_url=""),
            passthrough={"policyholder_name": ""},
        )
        # Empty passthrough is ignored — regex re-extracts.
        assert result.fields["policyholder_name"] == "Alice Homeowner"
        assert result.method["policyholder_name"] == "regex"

    def test_none_passthrough_ignored(self):
        text = "Policyholder Name: Alice Homeowner"
        result = extract(
            text,
            config=IntakeConfig(llm_base_url=""),
            passthrough={"policyholder_name": None},
        )
        assert result.fields["policyholder_name"] == "Alice Homeowner"


# ---------------------------------------------------------------------------
# LLM-assisted extraction (mocked)
# ---------------------------------------------------------------------------
class TestLLMAssistedExtraction:
    def test_llm_not_used_when_not_configured(self):
        text = "Some text without fields."
        result = extract(text, config=IntakeConfig(llm_base_url=""))
        assert result.llm_used is False

    def test_llm_called_for_missing_fields(self, monkeypatch):
        """When the LLM is configured, missing fields should be filled by it."""
        from claim_intake import extractor

        # Mock the LLM call to return a known value for policyholder_name.
        def fake_llm(field_name, *, ocr_text, config):
            if field_name == "policyholder_name":
                return "Alice From LLM", True
            return None, False

        monkeypatch.setattr(extractor, "_llm_extract_field", fake_llm)

        text = "Policy ID: HO-2024-001\nDate of Loss: 2026-03-14\nDescription: Storm damage."
        result = extract(
            text,
            config=IntakeConfig(llm_base_url="http://fake-lm-studio:1234/v1"),
        )
        assert result.llm_used is True
        assert result.fields["policyholder_name"] == "Alice From LLM"
        assert result.method["policyholder_name"] == "llm"

    def test_llm_failure_falls_back_to_missing(self, monkeypatch):
        from claim_intake import extractor

        def failing_llm(field_name, *, ocr_text, config):
            return None, False

        monkeypatch.setattr(extractor, "_llm_extract_field", failing_llm)

        text = "Policy ID: HO-2024-001"
        result = extract(
            text,
            config=IntakeConfig(llm_base_url="http://fake-lm-studio:1234/v1"),
        )
        # policyholder_name is missing — LLM was called but failed.
        assert "policyholder_name" not in result.fields
        assert result.method["policyholder_name"] == "missing"
        # llm_used is False because no field was actually populated by LLM.
        assert result.llm_used is False
