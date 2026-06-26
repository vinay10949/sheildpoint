"""
SP-500 — Fraud Detection Client
================================

High-level client that insurers use to interact with the cross-party
fraud detection network. Wraps the commitment service, the coordination
layer API client, and the non-membership prover into a single interface.

Usage (by the ClassifierAgent during the CLASSIFYING state)::

    client = FraudDetectionClient(
        insurer_id="shieldpoint",
        coordination_layer_url="https://coord.shieldpoint.example.com",
    )
    result = client.check_claim_uniqueness(
        claim_id="CLM-001",
        claimant_id=12345,
        date_of_loss="2026-03-14",
        location="geohash:dney0c2k",
        peril_type=1,  # wind
        amount=1250.00,
    )
    if not result.is_unique:
        # Duplicate detected — route to ESCALATING with fraud flag
        ...

In production, ``coordination_layer_url`` points to the real coordination
layer (SP-502). In tests, it can point to a local in-process mock or
to an :class:`InProcessCoordinationLayer` that uses an in-memory
:class:`SharedMerkleTree`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol
from urllib.parse import urljoin
import urllib.request
import urllib.error
import json

from .commitment import CommitmentService, Commitment
from .merkle_tree import SharedMerkleTree
from .non_membership_prover import (
    NonMembershipProver,
    NonMembershipResult,
    ProverMode,
)

logger = logging.getLogger("shieldpoint.fraud_detection.client")


@dataclass(frozen=True)
class FraudDetectionResult:
    """Result of a fraud-detection check for a single claim.

    Attributes
    ----------
    is_unique : bool
        True if the claim's commitment is NOT in the shared tree (no
        duplicate detected). False if a duplicate was found.
    commitment_value : str
        The Poseidon commitment generated for this claim.
    merkle_root : str
        The Merkle root the proof was generated against.
    proof : NonMembershipResult
        The full non-membership proof result (includes latency, proof
        object, and public signals).
    duplicate_insurer : str, optional
        If a duplicate was detected, the ID of the insurer that
        previously filed the claim (if known from the coordination layer).
    checked_at : float
        Unix timestamp of the check.
    """

    is_unique: bool
    commitment_value: str
    merkle_root: str
    proof: NonMembershipResult
    duplicate_insurer: Optional[str] = None
    checked_at: float = field(default_factory=time.time)


class CoordinationLayerClient(Protocol):
    """Protocol for coordination layer clients.

    Implemented by:
    - :class:`HTTPCoordinationLayerClient` — production, talks to the
      FastAPI service via HTTP.
    - :class:`InProcessCoordinationLayer` — tests / local dev, uses an
      in-memory SharedMerkleTree.
    """

    def get_root(self) -> str: ...
    def submit_commitment(self, commitment: str, insurer_id: str,
                          claim_id: str) -> dict[str, Any]: ...
    def get_non_membership_proof(self, commitment: str) -> dict[str, Any]: ...
    def get_membership_proof(self, commitment: str) -> Optional[dict[str, Any]]: ...


class InProcessCoordinationLayer:
    """In-process coordination layer for tests and local development.

    Uses an in-memory :class:`SharedMerkleTree` — no network calls.
    Implements the :class:`CoordinationLayerClient` protocol.
    """

    def __init__(self, depth: int = 20) -> None:
        self.tree = SharedMerkleTree(depth=depth)
        # Track which insurer submitted each commitment (for duplicate lookup)
        self._commitment_meta: dict[int, dict[str, str]] = {}

    def get_root(self) -> str:
        return str(self.tree.root)

    def submit_commitment(
        self, commitment: str, insurer_id: str, claim_id: str
    ) -> dict[str, Any]:
        c = int(commitment)
        inserted = self.tree.insert(c)
        if inserted:
            self._commitment_meta[c] = {"insurer_id": insurer_id, "claim_id": claim_id}
            return {
                "accepted": True,
                "new_root": str(self.tree.root),
                "duplicate": False,
            }
        else:
            # Duplicate — return the existing submitter info
            meta = self._commitment_meta.get(c, {})
            return {
                "accepted": False,
                "new_root": str(self.tree.root),
                "duplicate": True,
                "original_insurer": meta.get("insurer_id"),
                "original_claim_id": meta.get("claim_id"),
            }

    def get_non_membership_proof(self, commitment: str) -> dict[str, Any]:
        c = int(commitment)
        proof = self.tree.prove_non_membership(c)
        if proof is None:
            # Is a member — duplicate
            meta = self._commitment_meta.get(c, {})
            return {
                "is_member": True,
                "duplicate_insurer": meta.get("insurer_id"),
                "duplicate_claim_id": meta.get("claim_id"),
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

    def get_membership_proof(self, commitment: str) -> Optional[dict[str, Any]]:
        c = int(commitment)
        proof = self.tree.prove_membership(c)
        if proof is None:
            return None
        return proof.to_dict()


class HTTPCoordinationLayerClient:
    """HTTP client for the real coordination layer (SP-502).

    Talks to the FastAPI service via plain HTTP. In production, the
    server uses mutual TLS — this client should be configured with
    a client certificate via the ``cert`` parameter of
    :class:`urllib.request.Request` (or via the ``requests`` library
    if available).
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        cert_file: Optional[str] = None,
        key_file: Optional[str] = None,
        ca_file: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cert_file = cert_file
        self.key_file = key_file
        self.ca_file = ca_file

    def _request(
        self, method: str, path: str, body: Optional[dict] = None
    ) -> dict[str, Any]:
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        # SSL context for mTLS (production)
        import ssl
        ctx = ssl.create_default_context()
        if self.ca_file:
            ctx.load_verify_locations(cafile=self.ca_file)
        if self.cert_file and self.key_file:
            ctx.load_cert_chain(certfile=self.cert_file, keyfile=self.key_file)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"Coordination layer error {e.code}: {body}") from e

    def get_root(self) -> str:
        resp = self._request("GET", "/api/v1/root")
        return resp["root"]

    def submit_commitment(
        self, commitment: str, insurer_id: str, claim_id: str
    ) -> dict[str, Any]:
        return self._request("POST", "/api/v1/commitments", {
            "commitment": commitment,
            "insurer_id": insurer_id,
            "claim_id": claim_id,
        })

    def get_non_membership_proof(self, commitment: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/proofs/non-membership/{commitment}")

    def get_membership_proof(self, commitment: str) -> Optional[dict[str, Any]]:
        resp = self._request("GET", f"/api/v1/proofs/membership/{commitment}")
        if resp.get("is_member"):
            return resp.get("merkle_proof")
        return None


class FraudDetectionClient:
    """High-level client for the cross-party fraud detection network.

    Orchestrates:
    1. Commitment generation (local — never shares raw claim data).
    2. Coordination layer query for non-membership proof.
    3. Optional: submit the commitment to the shared tree after the
       claim is approved (so future duplicate checks can detect it).
    4. Local verification of the returned proof (defence in depth).

    Parameters
    ----------
    insurer_id : str
        Identifier for this insurer node (e.g. "shieldpoint").
    coordination_layer : CoordinationLayerClient
        Client for the coordination layer API. Use
        :class:`InProcessCoordinationLayer` for tests, or
        :class:`HTTPCoordinationLayerClient` for production.
    prover : NonMembershipProver, optional
        Local prover for re-verifying coordination-layer proofs. If
        None, a default AUTO-mode prover is created.
    auto_submit : bool
        If True, automatically submit the commitment to the shared
        tree after a successful non-membership check (so the claim
        can be detected as a duplicate by other insurers). Default True.
    """

    def __init__(
        self,
        *,
        insurer_id: str = "shieldpoint",
        coordination_layer: Optional[CoordinationLayerClient] = None,
        prover: Optional[NonMembershipProver] = None,
        auto_submit: bool = True,
    ) -> None:
        self.insurer_id = insurer_id
        self.coordination_layer = coordination_layer or InProcessCoordinationLayer()
        self.commitment_service = CommitmentService(insurer_id=insurer_id)
        self.prover = prover or NonMembershipProver(mode=ProverMode.AUTO)
        self.auto_submit = auto_submit

    def check_claim_uniqueness(
        self,
        *,
        claim_id: str,
        claimant_id: int,
        date_of_loss: str,
        location: str,
        peril_type: int,
        amount: float,
    ) -> FraudDetectionResult:
        """Check if a claim has been filed with any other insurer.

        This is the main entry point called by the ClassifierAgent
        during the CLASSIFYING state.

        Returns a :class:`FraudDetectionResult` with ``is_unique=True``
        if the claim is new (not a duplicate), or ``is_unique=False``
        if a duplicate was detected.
        """
        # Step 1: Generate the commitment locally
        commitment = self.commitment_service.create_commitment(
            claim_id=claim_id,
            claimant_id=claimant_id,
            date_of_loss=date_of_loss,
            location=location,
            peril_type=peril_type,
            amount=amount,
        )

        # Step 2: Query the coordination layer for a non-membership proof
        coord_result = self.coordination_layer.get_non_membership_proof(
            str(commitment.value)
        )

        is_unique = not coord_result.get("is_member", False)
        duplicate_insurer = coord_result.get("duplicate_insurer")

        # Step 3: Build the proof result
        if is_unique:
            # Parse the Merkle proof and re-verify locally
            merkle_data = coord_result.get("merkle_proof")
            if merkle_data:
                # Reconstruct the proof and verify locally (defence in depth)
                from .merkle_tree import NonMembershipProof
                local_proof = NonMembershipProof(
                    new_commitment=int(merkle_data["new_commitment"]),
                    left_neighbor=int(merkle_data["left_neighbor"]),
                    right_neighbor=int(merkle_data["right_neighbor"]),
                    left_is_sentinel=merkle_data["left_is_sentinel"],
                    right_is_sentinel=merkle_data["right_is_sentinel"],
                    left_path=[int(p) for p in merkle_data["left_path"]],
                    right_path=[int(p) for p in merkle_data["right_path"]],
                    left_path_indices=merkle_data["left_path_indices"],
                    right_path_indices=merkle_data["right_path_indices"],
                    root=int(merkle_data["root"]),
                )
                local_verified = SharedMerkleTree.verify_non_membership(local_proof)
                proof_result = NonMembershipResult(
                    verified=local_verified,
                    proof_type="coordination_layer",
                    statement=(
                        f"Non-membership proof {'VERIFIED' if local_verified else 'FAILED'} "
                        f"locally: commitment {commitment.value} is NOT in the "
                        f"shared tree (root={local_proof.root})."
                    ),
                    latency_ms=0.0,
                    verify_latency_ms=0.0,
                    proof={"type": "coordination_layer",
                           "merkle_proof": merkle_data},
                    public_signals={
                        "merkleRoot": str(local_proof.root),
                        "newCommitment": str(commitment.value),
                    },
                    merkle_root=str(local_proof.root),
                    new_commitment=str(commitment.value),
                )
            else:
                # Empty tree case
                proof_result = NonMembershipResult(
                    verified=True,
                    proof_type="empty_tree",
                    statement="Tree is empty — commitment is trivially unique.",
                    latency_ms=0.0,
                    verify_latency_ms=0.0,
                    proof={"type": "empty_tree"},
                    public_signals={
                        "merkleRoot": coord_result.get("root", "0"),
                        "newCommitment": str(commitment.value),
                    },
                    merkle_root=coord_result.get("root", "0"),
                    new_commitment=str(commitment.value),
                )

            # Step 4: Auto-submit to the shared tree (so future claims can detect this one)
            if self.auto_submit:
                self.coordination_layer.submit_commitment(
                    str(commitment.value),
                    insurer_id=self.insurer_id,
                    claim_id=claim_id,
                )
        else:
            # Duplicate detected
            proof_result = NonMembershipResult(
                verified=False,
                proof_type="duplicate_detected",
                statement=(
                    f"DUPLICATE DETECTED: commitment {commitment.value} is "
                    f"already in the shared tree. Previously filed by "
                    f"insurer={duplicate_insurer}."
                ),
                latency_ms=0.0,
                verify_latency_ms=0.0,
                proof={},
                public_signals={
                    "merkleRoot": coord_result.get("root", "0"),
                    "newCommitment": str(commitment.value),
                },
                merkle_root=coord_result.get("root", "0"),
                new_commitment=str(commitment.value),
            )

        return FraudDetectionResult(
            is_unique=is_unique,
            commitment_value=str(commitment.value),
            merkle_root=coord_result.get("root", "0"),
            proof=proof_result,
            duplicate_insurer=duplicate_insurer,
            checked_at=time.time(),
        )

    def get_commitment(self, claim_id: str) -> Optional[Commitment]:
        """Retrieve a previously-generated commitment by claim_id."""
        return self.commitment_service.get_commitment(claim_id)
