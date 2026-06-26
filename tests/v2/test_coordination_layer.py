"""
SP-502 — Coordination Layer API Tests
======================================

Tests for the inter-insurer coordination layer FastAPI service.

Run with::
    python -m pytest tests/v2/test_coordination_layer.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

# Ensure the coordination_layer package is on the path
_root = Path(__file__).resolve().parent.parent.parent
_coord_root = _root / "coordination_layer"
if str(_coord_root) not in sys.path:
    sys.path.insert(0, str(_coord_root.parent))

# Also ensure the zkp_circuit is on the path (dependency)
_zk_root = _root / "zkp_circuit"
if str(_zk_root) not in sys.path:
    sys.path.insert(0, str(_zk_root))


# Try to import FastAPI — skip all tests if not available
try:
    from fastapi.testclient import TestClient
    from coordination_layer.api import create_app, SQLiteCommitmentStore, MerkleTreeManager
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


pytestmark = pytest.mark.skipif(not _HAS_FASTAPI,
                                 reason="FastAPI not installed")


@pytest.fixture
def app_client():
    """Create a test app with an in-memory SQLite store."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = SQLiteCommitmentStore(db_path=db_path)
        app = create_app(store=store, depth=4)
        client = TestClient(app)
        yield client, store
    finally:
        os.unlink(db_path)


# ===========================================================================
# Health and Info Endpoints
# ===========================================================================
class TestHealthEndpoints:
    def test_root(self, app_client):
        client, _ = app_client
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "ShieldPoint Coordination Layer"

    def test_health(self, app_client):
        client, _ = app_client
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "leaf_count" in data
        assert "root" in data

    def test_get_root(self, app_client):
        client, _ = app_client
        resp = client.get("/api/v1/root")
        assert resp.status_code == 200
        data = resp.json()
        assert "root" in data
        assert "leaf_count" in data
        assert data["leaf_count"] == 0  # empty tree

    def test_stats(self, app_client):
        client, _ = app_client
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["leaf_count"] == 0
        assert data["capacity"] == 16  # 2^4
        assert data["utilization"] == 0.0


# ===========================================================================
# Commitment Submission
# ===========================================================================
class TestCommitmentSubmission:
    def test_submit_new_commitment(self, app_client):
        client, _ = app_client
        resp = client.post("/api/v1/commitments", json={
            "commitment": "12345",
            "insurer_id": "shieldpoint",
            "claim_id": "CLM-001",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is True
        assert data["duplicate"] is False
        assert data["leaf_count"] == 1

    def test_submit_duplicate_commitment(self, app_client):
        client, _ = app_client
        # First submission
        client.post("/api/v1/commitments", json={
            "commitment": "12345",
            "insurer_id": "shieldpoint",
            "claim_id": "CLM-001",
        })
        # Second submission (same commitment)
        resp = client.post("/api/v1/commitments", json={
            "commitment": "12345",
            "insurer_id": "allstate",
            "claim_id": "CLM-A-001",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is False
        assert data["duplicate"] is True
        assert data["original_insurer"] == "shieldpoint"

    def test_submit_invalid_commitment(self, app_client):
        """Non-numeric commitment should return 400."""
        client, _ = app_client
        resp = client.post("/api/v1/commitments", json={
            "commitment": "not-a-number",
            "insurer_id": "shieldpoint",
            "claim_id": "CLM-001",
        })
        assert resp.status_code == 400

    def test_root_updates_after_submission(self, app_client):
        client, _ = app_client
        root_before = client.get("/api/v1/root").json()["root"]
        client.post("/api/v1/commitments", json={
            "commitment": "12345",
            "insurer_id": "shieldpoint",
            "claim_id": "CLM-001",
        })
        root_after = client.get("/api/v1/root").json()["root"]
        assert root_before != root_after


# ===========================================================================
# Membership / Non-Membership Proofs
# ===========================================================================
class TestProofEndpoints:
    def test_membership_proof_for_existing(self, app_client):
        client, _ = app_client
        client.post("/api/v1/commitments", json={
            "commitment": "12345",
            "insurer_id": "shieldpoint",
            "claim_id": "CLM-001",
        })
        resp = client.get("/api/v1/proofs/membership/12345")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_member"] is True
        assert data["merkle_proof"] is not None
        assert data["insurer_id"] == "shieldpoint"

    def test_membership_proof_for_missing(self, app_client):
        client, _ = app_client
        resp = client.get("/api/v1/proofs/membership/99999")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_member"] is False

    def test_non_membership_proof_for_missing(self, app_client):
        client, _ = app_client
        client.post("/api/v1/commitments", json={
            "commitment": "100", "insurer_id": "sp", "claim_id": "C1",
        })
        client.post("/api/v1/commitments", json={
            "commitment": "300", "insurer_id": "sp", "claim_id": "C2",
        })
        resp = client.get("/api/v1/proofs/non-membership/200")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_member"] is False
        assert data["merkle_proof"] is not None

    def test_non_membership_proof_for_existing(self, app_client):
        """Non-membership for an EXISTING commitment → is_member=True (duplicate)."""
        client, _ = app_client
        client.post("/api/v1/commitments", json={
            "commitment": "12345", "insurer_id": "sp", "claim_id": "C1",
        })
        resp = client.get("/api/v1/proofs/non-membership/12345")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_member"] is True
        assert data["duplicate_insurer"] == "sp"


# ===========================================================================
# MerkleTreeManager Direct Tests
# ===========================================================================
class TestMerkleTreeManager:
    @pytest.mark.asyncio
    async def test_submit_and_retrieve(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = SQLiteCommitmentStore(db_path=db_path)
            mgr = MerkleTreeManager(store=store, depth=4)
            result = await mgr.submit_commitment(12345, "sp", "CLM-001")
            assert result["accepted"] is True
            assert result["leaf_count"] == 1

            # Duplicate
            result2 = await mgr.submit_commitment(12345, "other", "CLM-002")
            assert result2["accepted"] is False
            assert result2["duplicate"] is True
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_proof_generation_under_2_seconds(self):
        """AC: Proof generation (membership + non-membership) < 2 seconds."""
        import tempfile
        import time
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = SQLiteCommitmentStore(db_path=db_path)
            mgr = MerkleTreeManager(store=store, depth=10)
            # Insert 100 commitments
            for i in range(100):
                await mgr.submit_commitment(i * 100 + 50, "sp", f"CLM-{i}")

            start = time.perf_counter()
            await mgr.get_membership_proof(5050)
            await mgr.get_non_membership_proof(5075)
            elapsed = time.perf_counter() - start
            assert elapsed < 2.0, f"Proof generation took {elapsed:.2f}s"
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_notification_subscribers(self):
        """Submitter notifications are broadcast to subscribers."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            store = SQLiteCommitmentStore(db_path=db_path)
            mgr = MerkleTreeManager(store=store, depth=4)
            q = mgr.subscribe()
            await mgr.submit_commitment(12345, "sp", "CLM-001")
            # The notification should be in the queue
            event = await asyncio.wait_for(q.get(), timeout=1.0)
            assert event["event"] == "commitment_added"
            assert event["commitment"] == "12345"
        finally:
            os.unlink(db_path)
