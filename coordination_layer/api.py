"""
ShieldPoint Inter-Insurer Coordination Layer API (SP-502)
==========================================================

FastAPI service that serves as the lightweight communication backbone
for the cross-party fraud detection network.

Endpoints
---------
- ``GET  /api/v1/root``                         — current Merkle tree root
- ``POST /api/v1/commitments``                  — submit a new commitment
- ``GET  /api/v1/proofs/membership/{c}``        — membership proof for commitment c
- ``GET  /api/v1/proofs/non-membership/{c}``    — non-membership proof for commitment c
- ``GET  /api/v1/stats``                        — tree statistics
- ``WS   /api/v1/notifications``                — WebSocket for tree update notifications
- ``GET  /health``                              — health check
- ``GET  /``                                     — OpenAPI docs redirect

Security
--------
- **Mutual TLS**: all endpoints require a client certificate signed by
  the network's CA. Configure via the ``--ssl-certfile``, ``--ssl-keyfile``,
  and ``--ssl-ca-cert`` Uvicorn options (see ``start.sh``).
- **Data minimisation**: the service stores ONLY Poseidon hash values
  and Merkle tree metadata. No raw claim data ever enters this service.
- **Audit log**: every commitment submission is logged with the
  submitting insurer's certificate CN and timestamp.

Persistence
-----------
- **PostgreSQL** (production): stores commitments in the
  ``commitment_ledger`` table. Set ``DATABASE_URL`` env var.
- **SQLite** (development/tests): auto-fallback when ``DATABASE_URL``
  is unset. Uses a local ``coordination.db`` file.

Running
-------
Development (no mTLS, SQLite)::

    uvicorn coordination_layer.api:app --reload --port 8000

Production (mTLS, PostgreSQL)::

    uvicorn coordination_layer.api:app \
        --host 0.0.0.0 --port 8443 \
        --ssl-certfile /etc/shieldpoint/server.crt \
        --ssl-keyfile /etc/shieldpoint/server.key \
        --ssl-ca-certs /etc/shieldpoint/ca.crt
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("shieldpoint.coordination_layer")

# Sibling imports — the fraud_detection package
import sys
from pathlib import Path
_zk_root = Path(__file__).resolve().parent.parent / "zkp_circuit"
if str(_zk_root) not in sys.path:
    sys.path.insert(0, str(_zk_root))

from fraud_detection.merkle_tree import SharedMerkleTree  # noqa: E402
from fraud_detection.commitment import FIELD_PRIME  # noqa: E402

# Try to import PostgreSQL driver
try:
    import psycopg2  # type: ignore
    from psycopg2.extras import RealDictCursor  # type: ignore
    _HAS_POSTGRES = True
except ImportError:
    _HAS_POSTGRES = False


# ===========================================================================
# Storage backends
# ===========================================================================
class CommitmentStore:
    """Abstract commitment storage backend."""

    def insert(self, commitment: int, insurer_id: str, claim_id: str) -> bool:
        """Insert a commitment. Returns True if new, False if duplicate."""
        raise NotImplementedError

    def all_commitments(self) -> list[dict[str, Any]]:
        """Return all stored commitments (for tree reconstruction)."""
        raise NotImplementedError

    def find_by_value(self, commitment: int) -> Optional[dict[str, Any]]:
        """Look up a commitment by its value."""
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class SQLiteCommitmentStore(CommitmentStore):
    """SQLite-backed commitment storage (development / testing)."""

    def __init__(self, db_path: str = "coordination.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commitment_ledger (
                commitment TEXT PRIMARY KEY,
                insurer_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                submitted_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_insurer
            ON commitment_ledger(insurer_id)
        """)
        conn.commit()
        conn.close()

    def insert(self, commitment: int, insurer_id: str, claim_id: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO commitment_ledger VALUES (?, ?, ?, ?)",
                (str(commitment), insurer_id, claim_id, time.time()),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # duplicate
        finally:
            conn.close()

    def all_commitments(self) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM commitment_ledger ORDER BY commitment").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def find_by_value(self, commitment: int) -> Optional[dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM commitment_ledger WHERE commitment = ?",
            (str(commitment),),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM commitment_ledger").fetchone()[0]
        conn.close()
        return count


class PostgresCommitmentStore(CommitmentStore):
    """PostgreSQL-backed commitment storage (production)."""

    def __init__(self, database_url: str) -> None:
        if not _HAS_POSTGRES:
            raise RuntimeError("psycopg2 not installed")
        self.database_url = database_url
        self._init_db()

    def _init_db(self) -> None:
        conn = psycopg2.connect(self.database_url)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commitment_ledger (
                commitment TEXT PRIMARY KEY,
                insurer_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                submitted_at DOUBLE PRECISION NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_insurer
            ON commitment_ledger(insurer_id)
        """)
        conn.commit()
        conn.close()

    def insert(self, commitment: int, insurer_id: str, claim_id: str) -> bool:
        conn = psycopg2.connect(self.database_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO commitment_ledger VALUES (%s, %s, %s, %s)",
                    (str(commitment), insurer_id, claim_id, time.time()),
                )
            conn.commit()
            return True
        except psycopg2.IntegrityError:
            conn.rollback()
            return False
        finally:
            conn.close()

    def all_commitments(self) -> list[dict[str, Any]]:
        conn = psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
        rows = conn.execute(
            "SELECT * FROM commitment_ledger ORDER BY commitment"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def find_by_value(self, commitment: int) -> Optional[dict[str, Any]]:
        conn = psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
        row = conn.execute(
            "SELECT * FROM commitment_ledger WHERE commitment = %s",
            (str(commitment),),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def count(self) -> int:
        conn = psycopg2.connect(self.database_url)
        count = conn.execute("SELECT COUNT(*) FROM commitment_ledger").fetchone()[0]
        conn.close()
        return count


# ===========================================================================
# Merkle tree manager — wraps the tree + storage + notification
# ===========================================================================
class MerkleTreeManager:
    """Manages the shared Merkle tree with persistence and notifications.

    On startup, rebuilds the tree from the commitment ledger.
    All mutations go through this class to ensure the in-memory tree
    and the persistent store stay in sync.
    """

    def __init__(
        self,
        store: CommitmentStore,
        depth: int = 20,
    ) -> None:
        self.store = store
        self.tree = SharedMerkleTree(depth=depth)
        self._lock = asyncio.Lock()
        self._notification_subscribers: list[asyncio.Queue] = []
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        """Rebuild the in-memory tree from persistent storage."""
        records = self.store.all_commitments()
        for record in records:
            self.tree.insert(int(record["commitment"]))
        logger.info("Rebuilt Merkle tree with %d commitments", len(records))

    @property
    def root(self) -> int:
        return self.tree.root

    @property
    def leaf_count(self) -> int:
        return self.tree.leaf_count

    async def submit_commitment(
        self, commitment: int, insurer_id: str, claim_id: str
    ) -> dict[str, Any]:
        """Submit a new commitment. Returns the submission result."""
        async with self._lock:
            # Check for duplicate first (fast path)
            existing = self.store.find_by_value(commitment)
            if existing:
                return {
                    "accepted": False,
                    "duplicate": True,
                    "new_root": str(self.tree.root),
                    "original_insurer": existing["insurer_id"],
                    "original_claim_id": existing["claim_id"],
                }

            # Insert into persistent store
            inserted = self.store.insert(commitment, insurer_id, claim_id)
            if not inserted:
                # Race condition — another inserter beat us
                existing = self.store.find_by_value(commitment)
                return {
                    "accepted": False,
                    "duplicate": True,
                    "new_root": str(self.tree.root),
                    "original_insurer": existing["insurer_id"] if existing else None,
                    "original_claim_id": existing["claim_id"] if existing else None,
                }

            # Insert into in-memory tree
            self.tree.insert(commitment)
            new_root = self.tree.root

            # Notify subscribers
            await self._notify({
                "event": "commitment_added",
                "commitment": str(commitment),
                "insurer_id": insurer_id,
                "new_root": str(new_root),
                "leaf_count": self.tree.leaf_count,
                "timestamp": time.time(),
            })

            return {
                "accepted": True,
                "duplicate": False,
                "new_root": str(new_root),
                "leaf_count": self.tree.leaf_count,
            }

    async def get_membership_proof(self, commitment: int) -> dict[str, Any]:
        """Generate a membership proof."""
        async with self._lock:
            proof = self.tree.prove_membership(commitment)
            if proof is None:
                meta = self.store.find_by_value(commitment)
                return {
                    "is_member": False,
                    "merkle_proof": None,
                    "root": str(self.tree.root),
                }
            meta = self.store.find_by_value(commitment)
            return {
                "is_member": True,
                "merkle_proof": proof.to_dict(),
                "insurer_id": meta["insurer_id"] if meta else None,
                "claim_id": meta["claim_id"] if meta else None,
                "root": str(self.tree.root),
            }

    async def get_non_membership_proof(self, commitment: int) -> dict[str, Any]:
        """Generate a non-membership proof."""
        async with self._lock:
            proof = self.tree.prove_non_membership(commitment)
            if proof is None:
                # Is a member — duplicate
                meta = self.store.find_by_value(commitment)
                return {
                    "is_member": True,
                    "duplicate_insurer": meta["insurer_id"] if meta else None,
                    "duplicate_claim_id": meta["claim_id"] if meta else None,
                    "merkle_proof": None,
                    "root": str(self.tree.root),
                }
            return {
                "is_member": False,
                "duplicate_insurer": None,
                "duplicate_claim_id": None,
                "merkle_proof": proof.to_dict(),
                "root": str(self.tree.root),
            }

    # ------------------------------------------------------------------ #
    # WebSocket notification
    # ------------------------------------------------------------------ #
    def subscribe(self) -> asyncio.Queue:
        """Subscribe to tree update notifications. Returns a queue."""
        q: asyncio.Queue = asyncio.Queue()
        self._notification_subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._notification_subscribers:
            self._notification_subscribers.remove(q)

    async def _notify(self, event: dict[str, Any]) -> None:
        """Broadcast an event to all subscribers."""
        for q in self._notification_subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Notification queue full — dropping event")


# ===========================================================================
# Pydantic models (module-level for proper FastAPI registration)
# ===========================================================================
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
    from pydantic import BaseModel, Field

    class CommitmentSubmission(BaseModel):
        commitment: str = Field(..., description="Poseidon hash as decimal string")
        insurer_id: str = Field(..., description="Submitting insurer identifier")
        claim_id: str = Field(..., description="Internal claim ID at the insurer")

    class CommitmentResponse(BaseModel):
        accepted: bool
        duplicate: bool
        new_root: str
        leaf_count: Optional[int] = None
        original_insurer: Optional[str] = None
        original_claim_id: Optional[str] = None

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    BaseModel = object  # type: ignore


# ===========================================================================
# FastAPI app
# ===========================================================================
def create_app(
    *,
    store: Optional[CommitmentStore] = None,
    depth: int = 20,
) -> "FastAPI":
    """Create and configure the FastAPI application.

    Parameters
    ----------
    store : CommitmentStore, optional
        Storage backend. Defaults to SQLite (or PostgreSQL if DATABASE_URL is set).
    depth : int
        Merkle tree depth (default 20).
    """
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not installed. Install with: pip install fastapi")

    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
    from contextlib import asynccontextmanager

    # Resolve store
    if store is None:
        db_url = os.environ.get("DATABASE_URL")
        if db_url and _HAS_POSTGRES:
            store = PostgresCommitmentStore(db_url)
            logger.info("Using PostgreSQL store: %s", db_url)
        else:
            store = SQLiteCommitmentStore()
            logger.info("Using SQLite store")

    tree_manager = MerkleTreeManager(store=store, depth=depth)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Coordination layer started (depth=%d, leaves=%d)",
                    depth, tree_manager.leaf_count)
        yield
        logger.info("Coordination layer shutting down")

    app = FastAPI(
        title="ShieldPoint Inter-Insurer Coordination Layer",
        description=(
            "Cross-party ZKP fraud detection coordination layer. "
            "Stores ONLY cryptographic commitments — never raw claim data."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ---- Middleware: extract client identity from mTLS cert ----
    @app.middleware("http")
    async def extract_client_identity(request: Request, call_next):
        # In production with mTLS, the client cert CN is available via
        # request.scope["connection"].extra["ssl_object"].getpeercert()
        # We store it in request state for audit logging.
        cert = request.scope.get("connection", {}).get("extra", {}).get("ssl_object")
        if cert:
            peer_cert = cert.getpeercert()
            if peer_cert:
                subject = dict(x[0] for x in peer_cert.get("subject", []))
                request.state.insurer_id = subject.get("commonName", "unknown")
        else:
            request.state.insurer_id = "unknown"
        response = await call_next(request)
        return response

    # ---- Routes ----
    @app.get("/")
    async def root():
        return {
            "service": "ShieldPoint Coordination Layer",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "leaf_count": tree_manager.leaf_count,
            "root": str(tree_manager.root),
            "timestamp": time.time(),
        }

    @app.get("/api/v1/root")
    async def get_root():
        """Get the current Merkle tree root."""
        return {
            "root": str(tree_manager.root),
            "leaf_count": tree_manager.leaf_count,
            "depth": depth,
            "timestamp": time.time(),
        }

    @app.get("/api/v1/stats")
    async def get_stats():
        """Get tree statistics."""
        return {
            "leaf_count": tree_manager.leaf_count,
            "capacity": 2 ** depth,
            "utilization": tree_manager.leaf_count / (2 ** depth),
            "root": str(tree_manager.root),
            "depth": depth,
        }

    @app.post("/api/v1/commitments", response_model=CommitmentResponse)
    async def submit_commitment(submission: CommitmentSubmission, request: Request):
        """Submit a new commitment to the shared Merkle tree.

        The commitment must be a Poseidon hash (as a decimal string) in
        the BN128 scalar field. The coordination layer stores ONLY the
        commitment value — never the underlying claim data.

        Returns 200 with ``accepted=True`` if the commitment is new, or
        200 with ``accepted=False, duplicate=True`` if the commitment
        already exists (submitted by another insurer).
        """
        try:
            commitment_int = int(submission.commitment)
            if not (0 <= commitment_int < FIELD_PRIME):
                raise ValueError("Out of field range")
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid commitment: {e}")

        # Use the cert CN if available, else the submitted insurer_id
        insurer_id = getattr(request.state, "insurer_id", "unknown")
        if insurer_id == "unknown":
            insurer_id = submission.insurer_id

        result = await tree_manager.submit_commitment(
            commitment_int, insurer_id, submission.claim_id
        )
        return CommitmentResponse(**result)

    @app.get("/api/v1/proofs/membership/{commitment}")
    async def get_membership_proof(commitment: str):
        """Generate a membership proof for a commitment."""
        try:
            c = int(commitment)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid commitment format")
        result = await tree_manager.get_membership_proof(c)
        return result

    @app.get("/api/v1/proofs/non-membership/{commitment}")
    async def get_non_membership_proof(commitment: str):
        """Generate a non-membership proof for a commitment.

        If the commitment IS in the tree, returns ``is_member=true``
        with the original submitter info (duplicate detected).

        If the commitment is NOT in the tree, returns ``is_member=false``
        with the Merkle non-membership proof.
        """
        try:
            c = int(commitment)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid commitment format")
        result = await tree_manager.get_non_membership_proof(c)
        return result

    @app.websocket("/api/v1/notifications")
    async def notifications_ws(websocket: WebSocket):
        """WebSocket channel for tree update notifications.

        Clients receive a JSON event every time a new commitment is
        added to the tree. Event format::

            {
                "event": "commitment_added",
                "commitment": "<decimal_string>",
                "insurer_id": "<insurer>",
                "new_root": "<decimal_string>",
                "leaf_count": 42,
                "timestamp": 1234567890.123
            }
        """
        await websocket.accept()
        q = tree_manager.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    await websocket.send_json(event)
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    await websocket.send_json({"event": "keepalive"})
        except WebSocketDisconnect:
            pass
        finally:
            tree_manager.unsubscribe(q)

    return app


# Default app instance for uvicorn
# Wrapped in try/except so importing the module doesn't fail if the
# database backend isn't available (e.g. psycopg2 not installed in CI).
try:
    app = create_app()
except Exception as e:
    logger.warning("Failed to create default app: %s — using SQLite fallback", e)
    app = create_app(store=SQLiteCommitmentStore())
