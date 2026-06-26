"""API tests — FastAPI TestClient against the intake endpoints."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(test_config) -> Any:
    """A TestClient with the test_config's settings applied.

    We monkeypatch the IntakeConfig.from_env to return the test config
    so the API uses deterministic settings (no IMAP, no LLM).
    """
    from claim_intake import api, config
    original_from_env = config.IntakeConfig.from_env
    config.IntakeConfig.from_env = classmethod(lambda cls: test_config)
    # Also patch the api module's reference to from_env.
    api.IntakeConfig.from_env = classmethod(lambda cls: test_config)
    with TestClient(api.app) as c:
        yield c
    config.IntakeConfig.from_env = original_from_env
    api.IntakeConfig.from_env = original_from_env


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "store" in body
        assert "poller" in body
        assert "ocr_tesseract_available" in body


# ---------------------------------------------------------------------------
# Web claim intake
# ---------------------------------------------------------------------------
class TestWebClaimEndpoint:
    def test_submit_valid_claim(self, client):
        payload = {
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof shingles during storm.",
        }
        r = client.post("/intake/claims", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "accepted"
        assert body["accepted"] is True
        assert body["claim_id"].startswith("CLM-")
        assert body["claim"]["policyholder_name"] == "Alice Homeowner"
        assert body["latency_sec"] < 30.0

    def test_submit_incomplete_claim_routed_to_review(self, client):
        payload = {
            "policyholder_name": "Bob Driver",
            # policy_id missing
            "claim_type": "auto",
            # date_of_loss missing
            "damage_description": "Rear-end collision.",
        }
        r = client.post("/intake/claims", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "in_review"
        assert body["accepted"] is False
        assert body["review_queue_position"] is not None
        error_fields = {e["field"] for e in body["errors"]}
        assert "policy_id" in error_fields
        assert "date_of_loss" in error_fields

    def test_submit_with_request_id_round_trip(self, client):
        payload = {
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof shingles.",
            "request_id": "REQ-TEST-12345",
        }
        r = client.post("/intake/claims", json=payload)
        body = r.json()
        assert body["request_id"] == "REQ-TEST-12345"

    def test_get_claim_by_id(self, client):
        # First submit a claim.
        payload = {
            "policyholder_name": "Alice Homeowner",
            "policy_id": "HO-2024-001",
            "claim_type": "homeowners",
            "date_of_loss": "2026-03-14",
            "damage_description": "Wind damage to roof shingles.",
        }
        r = client.post("/intake/claims", json=payload)
        claim_id = r.json()["claim_id"]
        # Then retrieve it.
        r = client.get(f"/intake/claims/{claim_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["claim_id"] == claim_id
        assert body["claim"]["policyholder_name"] == "Alice Homeowner"

    def test_get_unknown_claim_returns_404(self, client):
        r = client.get("/intake/claims/CLM-DOES-NOT-EXIST")
        assert r.status_code == 404

    def test_list_claims(self, client):
        # Submit two claims.
        names = ["Alice Alpha", "Bob Beta"]
        for i, name in enumerate(names):
            client.post("/intake/claims", json={
                "policyholder_name": name,
                "policy_id": f"HO-2024-{i:03d}",
                "claim_type": "homeowners",
                "date_of_loss": "2026-03-14",
                "damage_description": "Wind damage to roof shingles.",
            })
        r = client.get("/intake/claims")
        assert r.status_code == 200
        body = r.json()
        assert len(body["claims"]) == 2
        assert body["accepted_count"] == 2


# ---------------------------------------------------------------------------
# Fax intake
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestFaxEndpoint:
    def test_upload_fax_pdf(self, client, make_fax_pdf):
        pdf_bytes = make_fax_pdf()
        r = client.post(
            "/intake/claims/fax",
            files={"file": ("claim.pdf", pdf_bytes, "application/pdf")},
            data={"fax_number": "+1-555-0100"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "accepted"
        assert body["claim"]["policyholder_name"] == "Alice Homeowner"
        assert body["claim"]["policy_id"] == "HO-2024-001"

    def test_upload_empty_pdf_returns_400(self, client):
        r = client.post(
            "/intake/claims/fax",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Email intake (manual trigger)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestEmailEndpoint:
    def test_submit_email_with_attachment(self, client, make_fax_pdf):
        pdf_bytes = make_fax_pdf()
        payload = {
            "sender": "adjuster@shieldpoint.example",
            "received_at": "2026-03-15T10:00:00Z",
            "subject": "New claim",
            "attachments": [{
                "filename": "claim.pdf",
                "mime_type": "application/pdf",
                "data_b64": base64.b64encode(pdf_bytes).decode(),
            }],
        }
        r = client.post("/intake/claims/email", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "accepted"
        assert body["claim"]["policyholder_name"] == "Alice Homeowner"


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------
class TestReviewQueue:
    def test_list_empty_queue(self, client):
        r = client.get("/review/queue")
        assert r.status_code == 200
        body = r.json()
        assert body["queue_depth"] == 0
        assert body["items"] == []

    def test_review_queue_populated_after_invalid_submission(self, client):
        client.post("/intake/claims", json={
            "policyholder_name": "Bob Driver",
            "damage_description": "Collision on the highway.",
        })
        r = client.get("/review/queue")
        body = r.json()
        assert body["queue_depth"] == 1
        assert body["items"][0]["status"] == "in_review"

    def test_get_review_item_by_id(self, client):
        post = client.post("/intake/claims", json={
            "policyholder_name": "Bob Driver",
            "damage_description": "Collision.",
        }).json()
        claim_id = post["claim_id"]
        r = client.get(f"/review/queue/{claim_id}")
        assert r.status_code == 200
        assert r.json()["claim_id"] == claim_id

    def test_resolve_review_accept(self, client):
        # Submit incomplete claim → goes to review.
        post = client.post("/intake/claims", json={
            "policyholder_name": "Bob Driver",
            "damage_description": "Collision on the highway.",
        }).json()
        claim_id = post["claim_id"]
        # Reviewer fills in missing fields.
        r = client.post(f"/review/queue/{claim_id}/resolve", json={
            "decision": "accept",
            "claim": {
                "policyholder_name": "Bob Driver",
                "policy_id": "AU-2024-015",
                "claim_type": "auto",
                "date_of_loss": "2026-04-02",
                "damage_description": "Rear-end collision on highway.",
            },
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "accepted"
        assert body["accepted"] is True

    def test_resolve_review_reject(self, client):
        post = client.post("/intake/claims", json={
            "damage_description": "Something happened.",
        }).json()
        claim_id = post["claim_id"]
        r = client.post(f"/review/queue/{claim_id}/resolve", json={
            "decision": "reject",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "rejected"

    def test_pop_next_review_fifo(self, client):
        # Submit 3 incomplete claims.
        names = ["Alice Alpha", "Bob Beta", "Carol Gamma"]
        ids = []
        for name in names:
            r = client.post("/intake/claims", json={
                "policyholder_name": name,
                "damage_description": "Short desc.",
            })
            ids.append(r.json()["claim_id"])
        # Pop next — should return the FIRST submitted.
        r = client.post("/review/queue/next")
        assert r.status_code == 200
        assert r.json()["claim_id"] == ids[0]
        # Pop again — second.
        r = client.post("/review/queue/next")
        assert r.json()["claim_id"] == ids[1]

    def test_pop_next_empty_returns_404(self, client):
        r = client.post("/review/queue/next")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Poller control
# ---------------------------------------------------------------------------
class TestPollerControl:
    def test_poller_status(self, client):
        r = client.get("/email/poller/status")
        assert r.status_code == 200
        body = r.json()
        assert "running" in body
        assert "poll_count" in body

    def test_poll_now_when_imap_disabled(self, client):
        r = client.post("/email/poller/poll-now")
        assert r.status_code == 200
        body = r.json()
        # IMAP disabled — no messages found, but no error.
        assert body["messages_found"] == 0
        assert body["errors"] == 0
