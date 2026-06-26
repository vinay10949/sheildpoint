"""
SP-501 — Non-Membership Prover (Python wrapper)
================================================

Wraps the Circom non-membership circuit for cross-party fraud detection.
Auto-detects compiled Groth16 artifacts (``non_membership_final.zkey``
and ``verification_key.json``); falls back to a deterministic stub
prover when artifacts are absent (for local development and CI).

Modes
-----
- ``ProverMode.GROTH16`` — Uses the real Circom + snarkjs prover via
  subprocess. Requires the circuit to be compiled and the trusted setup
  to be complete. Produces real zk-SNARK proofs verifiable in < 10ms.
- ``ProverMode.STUB`` — Deterministic stub that simulates the proof
  generation by running the Merkle tree verification in Python. The
  "proof" is a JSON blob containing the witness data. Used for
  development and CI where Circom isn't installed.
- ``ProverMode.AUTO`` (default) — Tries Groth16 first, falls back to
  stub if artifacts or tools are missing.

Performance (target)
--------------------
- Proof generation: < 10 seconds on CPU (Groth16)
- Verification: < 10ms (Groth16 constant time)
- Stub mode: < 100ms for both (Python-only)
"""

from __future__ import annotations

import enum
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .commitment import FIELD_PRIME
from .merkle_tree import NonMembershipProof, SharedMerkleTree

logger = logging.getLogger("shieldpoint.fraud_detection.prover")


class ProverMode(str, enum.Enum):
    """Prover operating mode."""
    AUTO = "auto"      # Try Groth16, fall back to stub
    GROTH16 = "groth16"  # Real Circom + snarkjs proofs only
    STUB = "stub"      # Deterministic Python stub (no Circom)


@dataclass(frozen=True)
class NonMembershipResult:
    """Result of a non-membership proof generation + verification.

    Attributes
    ----------
    verified : bool
        True if the commitment is proven NOT to be in the tree.
        False if the commitment IS in the tree (duplicate detected!).
    proof_type : str
        "groth16", "stub", or "empty_tree".
    statement : str
        Human-readable proof statement.
    latency_ms : float
        Time to generate the proof, in milliseconds.
    verify_latency_ms : float
        Time to verify the proof, in milliseconds.
    proof : dict
        The proof object (format depends on proof_type).
    public_signals : dict
        Public inputs to the proof (merkleRoot, newCommitment).
    merkle_root : str
        The Merkle root the proof was generated against.
    new_commitment : str
        The commitment checked for non-membership.
    """

    verified: bool
    proof_type: str
    statement: str
    latency_ms: float
    verify_latency_ms: float
    proof: dict[str, Any] = field(default_factory=dict)
    public_signals: dict[str, Any] = field(default_factory=dict)
    merkle_root: str = ""
    new_commitment: str = ""


class NonMembershipProver:
    """Generate and verify non-membership proofs against a Merkle tree.

    Parameters
    ----------
    circuit_dir : Path, optional
        Directory containing the compiled circuit artifacts
        (``non_membership_js/``, ``non_membership_final.zkey``,
        ``verification_key.json``). Defaults to the standard location
        under ``zkp_circuit/build`` and ``zkp_circuit/keys``.
    mode : ProverMode
        Operating mode. ``AUTO`` tries Groth16 first, falls back to stub.
    """

    # Default tree depth — must match the Circom circuit's `depth` parameter
    DEFAULT_DEPTH = 20

    def __init__(
        self,
        *,
        circuit_dir: Optional[Path] = None,
        mode: ProverMode = ProverMode.AUTO,
        depth: int = DEFAULT_DEPTH,
    ) -> None:
        self.mode = mode
        self.depth = depth
        # Resolve circuit artifact paths
        zk_root = Path(__file__).resolve().parent.parent
        self.build_dir = circuit_dir or (zk_root / "build" / "non_membership")
        self.keys_dir = zk_root / "keys"
        self.wasm_path = self.build_dir / "non_membership_js" / "non_membership.wasm"
        self.zkey_path = self.keys_dir / "non_membership_final.zkey"
        self.vkey_path = self.keys_dir / "non_membership_verification_key.json"
        # Detect available tools
        self._snarkjs = shutil.which("snarkjs")
        self._node = shutil.which("node")
        # Determine effective mode
        self._effective_mode = self._resolve_mode()

    def _resolve_mode(self) -> ProverMode:
        """Resolve the effective mode based on available tools/artifacts."""
        if self.mode == ProverMode.STUB:
            return ProverMode.STUB
        if self.mode == ProverMode.GROTH16:
            if not self._can_groth16():
                raise RuntimeError(
                    "Groth16 mode requested but artifacts/tools missing. "
                    f"Need: wasm={self.wasm_path}, zkey={self.zkey_path}, "
                    f"snarkjs={self._snarkjs}"
                )
            return ProverMode.GROTH16
        # AUTO mode
        if self._can_groth16():
            return ProverMode.GROTH16
        logger.info(
            "Groth16 artifacts not available — falling back to stub prover. "
            "Run `make compile-fraud fraud-trusted-setup` to enable real proofs."
        )
        return ProverMode.STUB

    def _can_groth16(self) -> bool:
        """Check if all Groth16 prerequisites are available."""
        return (
            self._snarkjs is not None
            and self.wasm_path.exists()
            and self.zkey_path.exists()
            and self.vkey_path.exists()
        )

    @property
    def effective_mode(self) -> ProverMode:
        return self._effective_mode

    # ------------------------------------------------------------------ #
    # Proof generation
    # ------------------------------------------------------------------ #
    def prove(
        self,
        tree: SharedMerkleTree,
        commitment: int,
    ) -> NonMembershipResult:
        """Generate a non-membership proof for ``commitment`` against ``tree``.

        If the commitment IS in the tree, returns a result with
        ``verified=False`` (duplicate detected — fraud flag).

        If the tree is empty, returns a trivial "empty_tree" proof
        (no circuit invocation needed).
        """
        if tree.leaf_count == 0:
            return self._empty_tree_proof(tree, commitment)

        start = time.perf_counter()
        # Generate the Merkle non-membership proof
        merkle_proof = tree.prove_non_membership(commitment)
        if merkle_proof is None:
            # Commitment IS in the tree — duplicate detected
            elapsed = (time.perf_counter() - start) * 1000
            return NonMembershipResult(
                verified=False,
                proof_type="duplicate_detected",
                statement=(
                    f"DUPLICATE DETECTED: commitment {commitment} is already "
                    f"in the shared Merkle tree (root={tree.root}). "
                    f"This claim has been filed with another insurer."
                ),
                latency_ms=elapsed,
                verify_latency_ms=0.0,
                proof={},
                public_signals={
                    "merkleRoot": str(tree.root),
                    "newCommitment": str(commitment),
                },
                merkle_root=str(tree.root),
                new_commitment=str(commitment),
            )

        # Generate the ZK proof (Groth16 or stub)
        if self._effective_mode == ProverMode.GROTH16:
            result = self._prove_groth16(merkle_proof, start)
        else:
            result = self._prove_stub(merkle_proof, start)
        return result

    def _empty_tree_proof(
        self, tree: SharedMerkleTree, commitment: int
    ) -> NonMembershipResult:
        """Trivial proof for an empty tree (no circuit needed)."""
        return NonMembershipResult(
            verified=True,
            proof_type="empty_tree",
            statement=(
                "Tree is empty — commitment is trivially not a member. "
                f"(newCommitment={commitment}, root={tree.root})"
            ),
            latency_ms=0.1,
            verify_latency_ms=0.1,
            proof={"type": "empty_tree"},
            public_signals={
                "merkleRoot": str(tree.root),
                "newCommitment": str(commitment),
            },
            merkle_root=str(tree.root),
            new_commitment=str(commitment),
        )

    def _prove_stub(
        self, merkle_proof: NonMembershipProof, start: float
    ) -> NonMembershipResult:
        """Stub prover: verify the Merkle proof in Python and package it."""
        # Verify the Merkle non-membership proof
        verified = SharedMerkleTree.verify_non_membership(merkle_proof)
        elapsed_gen = (time.perf_counter() - start) * 1000

        # Verify latency (stub: just re-check)
        v_start = time.perf_counter()
        SharedMerkleTree.verify_non_membership(merkle_proof)
        elapsed_ver = (time.perf_counter() - v_start) * 1000

        return NonMembershipResult(
            verified=verified,
            proof_type="stub",
            statement=(
                f"Non-membership proof {'VERIFIED' if verified else 'FAILED'}: "
                f"commitment {merkle_proof.new_commitment} is "
                f"{'NOT' if verified else 'possibly'} in the shared tree "
                f"(root={merkle_proof.root})."
            ),
            latency_ms=elapsed_gen,
            verify_latency_ms=elapsed_ver,
            proof={
                "type": "stub_non_membership",
                "merkle_proof": merkle_proof.to_dict(),
            },
            public_signals={
                "merkleRoot": str(merkle_proof.root),
                "newCommitment": str(merkle_proof.new_commitment),
            },
            merkle_root=str(merkle_proof.root),
            new_commitment=str(merkle_proof.new_commitment),
        )

    def _prove_groth16(
        self, merkle_proof: NonMembershipProof, start: float
    ) -> NonMembershipResult:
        """Real Groth16 prover via snarkjs subprocess."""
        # Build the witness input
        inputs = self._build_witness_input(merkle_proof)
        # Write input JSON
        input_file = self.build_dir / "input.json"
        input_file.parent.mkdir(parents=True, exist_ok=True)
        input_file.write_text(json.dumps(inputs))

        # Generate witness
        witness_file = self.build_dir / "witness.wtns"
        witness_gen = self.build_dir / "non_membership_js" / "generate_witness.js"
        try:
            subprocess.run(
                [self._node, str(witness_gen), str(self.wasm_path),
                 str(input_file), str(witness_file)],
                check=True, capture_output=True, timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error("Witness generation failed: %s", e)
            return self._prove_stub(merkle_proof, start)

        # Generate proof
        proof_file = self.build_dir / "proof.json"
        public_file = self.build_dir / "public.json"
        try:
            subprocess.run(
                [self._snarkjs, "groth16", "prove",
                 str(self.zkey_path), str(witness_file),
                 str(proof_file), str(public_file)],
                check=True, capture_output=True, timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error("Proof generation failed: %s", e)
            return self._prove_stub(merkle_proof, start)

        elapsed_gen = (time.perf_counter() - start) * 1000

        proof = json.loads(proof_file.read_text())
        public_signals = json.loads(public_file.read_text())

        # Verify the proof
        v_start = time.perf_counter()
        verified = self._verify_groth16(proof, public_signals)
        elapsed_ver = (time.perf_counter() - v_start) * 1000

        return NonMembershipResult(
            verified=verified,
            proof_type="groth16",
            statement=(
                f"Non-membership proof {'VERIFIED' if verified else 'FAILED'}: "
                f"commitment {merkle_proof.new_commitment} is "
                f"{'NOT' if verified else 'possibly'} in the shared tree "
                f"(root={merkle_proof.root})."
            ),
            latency_ms=elapsed_gen,
            verify_latency_ms=elapsed_ver,
            proof=proof,
            public_signals={
                "merkleRoot": public_signals[0] if len(public_signals) > 0 else str(merkle_proof.root),
                "newCommitment": public_signals[1] if len(public_signals) > 1 else str(merkle_proof.new_commitment),
            },
            merkle_root=str(merkle_proof.root),
            new_commitment=str(merkle_proof.new_commitment),
        )

    def _build_witness_input(
        self, merkle_proof: NonMembershipProof
    ) -> dict[str, Any]:
        """Build the Circom witness input JSON from a Merkle non-membership proof."""
        # Pad paths to depth length (should already be correct, but be defensive)
        depth = self.depth
        left_path = list(merkle_proof.left_path) + [0] * (depth - len(merkle_proof.left_path))
        right_path = list(merkle_proof.right_path) + [0] * (depth - len(merkle_proof.right_path))
        left_idx = list(merkle_proof.left_path_indices) + [0] * (depth - len(merkle_proof.left_path_indices))
        right_idx = list(merkle_proof.right_path_indices) + [0] * (depth - len(merkle_proof.right_path_indices))

        return {
            "merkleRoot": str(merkle_proof.root),
            "newCommitment": str(merkle_proof.new_commitment),
            "leftNeighbor": str(merkle_proof.left_neighbor),
            "rightNeighbor": str(merkle_proof.right_neighbor),
            "leftPath": [str(p) for p in left_path[:depth]],
            "rightPath": [str(p) for p in right_path[:depth]],
            "leftPathIndices": left_idx[:depth],
            "rightPathIndices": right_idx[:depth],
            "leftIsSentinel": "1" if merkle_proof.left_is_sentinel else "0",
            "rightIsSentinel": "1" if merkle_proof.right_is_sentinel else "0",
        }

    def _verify_groth16(self, proof: dict, public_signals: list) -> bool:
        """Verify a Groth16 proof via snarkjs."""
        if not self._snarkjs or not self.vkey_path.exists():
            return False
        try:
            result = subprocess.run(
                [self._snarkjs, "groth16", "verify",
                 str(self.vkey_path), json.dumps(public_signals), json.dumps(proof)],
                check=True, capture_output=True, timeout=10,
            )
            return b"OK!" in result.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    # ------------------------------------------------------------------ #
    # Standalone verification (no tree needed)
    # ------------------------------------------------------------------ #
    def verify(self, result: NonMembershipResult) -> bool:
        """Re-verify a previously-generated non-membership proof."""
        if result.proof_type == "empty_tree":
            return True
        if result.proof_type == "duplicate_detected":
            return False  # duplicate — not a valid non-membership
        if result.proof_type == "stub":
            # Re-verify the embedded Merkle proof
            merkle_data = result.proof.get("merkle_proof", {})
            if not merkle_data:
                return False
            proof = NonMembershipProof(
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
            return SharedMerkleTree.verify_non_membership(proof)
        if result.proof_type == "groth16":
            public = [
                result.public_signals.get("merkleRoot"),
                result.public_signals.get("newCommitment"),
            ]
            return self._verify_groth16(result.proof, public)
        return False
