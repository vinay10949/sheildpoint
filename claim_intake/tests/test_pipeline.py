"""End-to-end pipeline tests — web, email, fax."""

from __future__ import annotations

import time

import pytest

from claim_intake.config import IntakeConfig
from claim_intake.pipeline import intake_email_claim, intake_fax_claim, intake_web_claim
from claim_intake.schemas import ClaimStatus, IntakeSource
from claim_intake.store import get_accepted, get_review, list_review, store_stats


# ---------------------------------------------------------------------------
# Web claim intake
# ---------------------------------------------------------------------------
class TestWebClaimIntake:
    def test_valid_web_claim_accepted(self, valid_web_submission, test_config):
        result = intake_web_claim(valid_web_submission, config=test_config)
        assert result.status == ClaimStatus.ACCEPTED
        assert result.accepted is True
        assert result.claim is not None
        assert result.claim.policyholder_name == "Alice Homeowner"
        assert result.claim.policy_id == "HO-2024-001"
        assert result.claim_id.startswith("CLM-")
        # Should be persisted in the accepted store.
        rec = get_accepted(result.claim_id)
        assert rec is not None
        assert rec["status"] == ClaimStatus.ACCEPTED

    def test_incomplete_web_claim_routed_to_review(self, incomplete_web_submission, test_config):
        result = intake_web_claim(incomplete_web_submission, config=test_config)
        assert result.status == ClaimStatus.IN_REVIEW
        assert result.accepted is False
        assert result.claim is None
        assert len(result.errors) > 0
        # Missing fields should be flagged with field-level errors.
        error_fields = {e.field for e in result.errors}
        assert "policy_id" in error_fields
        assert "date_of_loss" in error_fields
        # Should be in the review queue.
        item = get_review(result.claim_id)
        assert item is not None
        assert item.status == ClaimStatus.IN_REVIEW
        assert item.source == IntakeSource.WEB

    def test_empty_web_claim_rejected(self, empty_web_submission, test_config):
        result = intake_web_claim(empty_web_submission, config=test_config)
        # 4+ missing required fields → rejected (max_missing_fields=5 default)
        assert result.status == ClaimStatus.REJECTED
        assert result.accepted is False

    def test_web_claim_latency_under_30s(self, valid_web_submission, test_config):
        """SP-203 AC: web intake < 30s end-to-end."""
        start = time.monotonic()
        result = intake_web_claim(valid_web_submission, config=test_config)
        elapsed = time.monotonic() - start
        assert result.status == ClaimStatus.ACCEPTED
        assert elapsed < 30.0, f"Web intake took {elapsed:.2f}s (AC: <30s)"
        assert result.latency_sec < 30.0

    def test_web_claim_with_partial_fields_extracts_from_description(self, test_config):
        """Portal submits policyholder_name + damage_description only —
        extractor should infer claim_type from the description."""
        from claim_intake.schemas import WebClaimSubmission
        sub = WebClaimSubmission(
            policyholder_name="Bob Driver",
            policy_id="AU-2024-015",
            # claim_type missing — extractor should infer "auto" from "collision"
            date_of_loss="2026-04-02",
            damage_description="Rear-end collision on the highway. Front bumper crushed.",
        )
        result = intake_web_claim(sub, config=test_config)
        assert result.status == ClaimStatus.ACCEPTED
        assert result.claim.claim_type.value == "auto"

    def test_web_claim_generates_unique_claim_ids(self, valid_web_submission, test_config):
        r1 = intake_web_claim(valid_web_submission, config=test_config)
        r2 = intake_web_claim(valid_web_submission, config=test_config)
        assert r1.claim_id != r2.claim_id


# ---------------------------------------------------------------------------
# Fax claim intake (full OCR pipeline)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestFaxClaimIntake:
    def test_valid_fax_accepted(self, make_fax_submission, test_config):
        """A clean fax PDF should be OCR'd, extracted, validated, accepted."""
        sub = make_fax_submission()
        result = intake_fax_claim(sub, config=test_config)
        assert result.status == ClaimStatus.ACCEPTED, (
            f"Expected ACCEPTED, got {result.status}. Errors: {result.errors}"
        )
        assert result.claim is not None
        assert result.claim.policyholder_name == "Alice Homeowner"
        assert result.claim.policy_id == "HO-2024-001"
        assert result.claim.claim_type.value == "homeowners"
        assert result.claim.date_of_loss == "2026-03-14"

    def test_fax_with_missing_field_routed_to_review(self, make_fax_submission, test_config):
        """Fax missing the policy_id label should go to review."""
        sub = make_fax_submission(policy_id="")  # blank policy_id line
        result = intake_fax_claim(sub, config=test_config)
        # The fax has no policy_id text now — should fail validation.
        assert result.status in (ClaimStatus.IN_REVIEW, ClaimStatus.REJECTED)
        assert result.accepted is False

    def test_fax_latency_under_2min(self, make_fax_submission, test_config):
        """SP-203 AC: OCR-based intake < 2 minutes."""
        sub = make_fax_submission()
        start = time.monotonic()
        result = intake_fax_claim(sub, config=test_config)
        elapsed = time.monotonic() - start
        assert elapsed < 120.0, f"Fax intake took {elapsed:.2f}s (AC: <2min)"
        assert result.latency_sec < 120.0


# ---------------------------------------------------------------------------
# Email claim intake (OCR on attachments)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestEmailClaimIntake:
    def test_valid_email_with_pdf_attachment_accepted(self, make_email_submission, test_config):
        sub = make_email_submission()
        result = intake_email_claim(sub, config=test_config)
        assert result.status == ClaimStatus.ACCEPTED, (
            f"Expected ACCEPTED, got {result.status}. Errors: {result.errors}"
        )
        assert result.claim is not None
        assert result.claim.policyholder_name == "Alice Homeowner"
        assert result.claim.policy_id == "HO-2024-001"

    def test_email_with_no_attachments_routed_to_review(self, test_config):
        from claim_intake.schemas import EmailClaimSubmission
        sub = EmailClaimSubmission(
            sender="adjuster@shieldpoint.example",
            received_at="2026-03-15T10:00:00Z",
            subject="Claim — no attachments",
            attachments=[],
        )
        result = intake_email_claim(sub, config=test_config)
        # No attachments → no OCR text → no fields → routed to review.
        assert result.status in (ClaimStatus.IN_REVIEW, ClaimStatus.REJECTED)
        assert result.accepted is False


# ---------------------------------------------------------------------------
# Store interactions
# ---------------------------------------------------------------------------
class TestStoreInteractions:
    def test_accepted_claim_persisted(self, valid_web_submission, test_config):
        result = intake_web_claim(valid_web_submission, config=test_config)
        stats = store_stats()
        assert stats["accepted_count"] == 1

    def test_review_queue_populated(self, incomplete_web_submission, test_config):
        result = intake_web_claim(incomplete_web_submission, config=test_config)
        items = list_review()
        assert len(items) == 1
        assert items[0].claim_id == result.claim_id

    def test_review_queue_filters_by_status(self, empty_web_submission, test_config):
        intake_web_claim(empty_web_submission, config=test_config)
        rejected_only = list_review(status=ClaimStatus.REJECTED)
        in_review_only = list_review(status=ClaimStatus.IN_REVIEW)
        assert len(rejected_only) == 1
        assert len(in_review_only) == 0


# ---------------------------------------------------------------------------
# Review resolution
# ---------------------------------------------------------------------------
class TestReviewResolution:
    def test_resolve_accept_promotes_to_accepted(self, incomplete_web_submission, test_config):
        from claim_intake.store import resolve_review
        result = intake_web_claim(incomplete_web_submission, config=test_config)
        assert result.status == ClaimStatus.IN_REVIEW

        # Reviewer fills in the missing fields and resolves.
        from claim_intake.schemas import StandardClaim, ClaimType
        completed = StandardClaim(
            policyholder_name="Bob Driver",
            policy_id="AU-2024-015",
            claim_type=ClaimType.AUTO,
            date_of_loss="2026-04-02",
            damage_description="Rear-end collision on highway.",
        )
        resolved = resolve_review(
            result.claim_id, accepted_claim=completed,
            source=IntakeSource.WEB, request_id=None,
        )
        assert resolved.status == ClaimStatus.ACCEPTED
        assert resolved.accepted is True

    def test_resolve_reject_marks_rejected(self, empty_web_submission, test_config):
        from claim_intake.store import resolve_review
        result = intake_web_claim(empty_web_submission, config=test_config)
        resolved = resolve_review(
            result.claim_id, accepted_claim=None,
            source=IntakeSource.WEB, request_id=None,
        )
        assert resolved.status == ClaimStatus.REJECTED
        assert resolved.accepted is False
