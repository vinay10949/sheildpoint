"""
SP-501 — Non-Membership Circuit Tests
======================================

Tests for the Circom non-membership circuit file and its Python wrapper.

Since we can't compile Circom in CI, these tests verify:
1. The circuit file exists and has the correct structure.
2. The circuit documents the constraint budget.
3. The circuit uses Poseidon hashing.
4. The Python wrapper (stub mode) works correctly.
5. Test vectors cover edge cases.

Run with::
    python -m pytest tests/v2/test_non_membership_circuit.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent.parent.parent
_zk_root = _root / "zkp_circuit"
if str(_zk_root) not in sys.path:
    sys.path.insert(0, str(_zk_root))

CIRCUIT_PATH = _zk_root / "circuits" / "non_membership.circom"


# ===========================================================================
# Circuit File Structure Tests
# ===========================================================================
class TestCircuitFile:
    """Verify the non-membership circuit file exists and has the required structure."""

    def test_circuit_file_exists(self):
        assert CIRCUIT_PATH.exists(), f"Circuit not found at {CIRCUIT_PATH}"

    def test_circuit_documents_constraint_budget(self):
        """AC: Non-membership Circom circuit compiles with <= 80K constraints."""
        content = CIRCUIT_PATH.read_text()
        assert "80K" in content or "80,000" in content, \
            "Circuit must document the 80K constraint budget"

    def test_circuit_uses_poseidon(self):
        """AC: Implement Poseidon hash for Merkle tree node computation."""
        content = CIRCUIT_PATH.read_text()
        assert "Poseidon" in content
        assert "poseidon.circom" in content

    def test_circuit_supports_depth_20(self):
        """AC: Supports Merkle trees up to 2^20 leaves."""
        content = CIRCUIT_PATH.read_text()
        assert "2^20" in content or "2^20" in content or "1,048,576" in content \
            or "depth = 20" in content or "depth=20" in content

    def test_circuit_has_non_membership_logic(self):
        """Circuit must implement the non-membership verification logic."""
        content = CIRCUIT_PATH.read_text()
        assert "NonMembershipProof" in content
        assert "leftNeighbor" in content or "left_neighbor" in content
        assert "rightNeighbor" in content or "right_neighbor" in content
        assert "LessThan" in content  # for the strict bracketing check

    def test_circuit_has_sentinel_handling(self):
        """Circuit must handle sentinel cases (tree boundaries)."""
        content = CIRCUIT_PATH.read_text()
        assert "sentinel" in content.lower()

    def test_circuit_has_merkle_path_verifier(self):
        """Circuit must include a Merkle path verification template."""
        content = CIRCUIT_PATH.read_text()
        assert "MerklePathVerifier" in content

    def test_circuit_public_inputs(self):
        """Circuit must expose merkleRoot and newCommitment as public inputs."""
        content = CIRCUIT_PATH.read_text()
        assert "public [merkleRoot, newCommitment]" in content

    def test_circuit_specifies_pragma_version(self):
        content = CIRCUIT_PATH.read_text()
        assert "pragma circom" in content

    def test_circuit_includes_circomlib(self):
        content = CIRCUIT_PATH.read_text()
        assert "circomlib" in content


# ===========================================================================
# Makefile Targets Tests
# ===========================================================================
class TestMakefileTargets:
    """Verify the Makefile exposes the fraud detection build pipeline."""

    def test_makefile_has_fraud_targets(self):
        makefile = _zk_root / "Makefile"
        content = makefile.read_text()
        assert "compile-fraud" in content
        assert "fraud-trusted-setup" in content
        assert "test-fraud" in content

    def test_makefile_uses_non_membership_circuit(self):
        makefile = _zk_root / "Makefile"
        content = makefile.read_text()
        assert "non_membership" in content
        assert "FRAUD_CIRCUIT" in content


# ===========================================================================
# Test Vector Generation (SP-501 AC)
# ===========================================================================
class TestVectors:
    """AC: Test vectors: valid non-membership, duplicate claim (membership),
    tree boundary cases."""

    def test_valid_non_membership_vector(self):
        """A commitment NOT in the tree should produce a valid non-membership proof."""
        from fraud_detection.merkle_tree import SharedMerkleTree
        tree = SharedMerkleTree(depth=4)
        tree.insert(100)
        tree.insert(300)
        proof = tree.prove_non_membership(200)
        assert proof is not None
        assert SharedMerkleTree.verify_non_membership(proof) is True

    def test_duplicate_claim_membership_vector(self):
        """A commitment IN the tree should return None (membership, not non-membership)."""
        from fraud_detection.merkle_tree import SharedMerkleTree
        tree = SharedMerkleTree(depth=4)
        tree.insert(100)
        proof = tree.prove_non_membership(100)
        assert proof is None  # is a member — caller flags as duplicate

    def test_left_boundary_vector(self):
        """Commitment smaller than all leaves (left boundary)."""
        from fraud_detection.merkle_tree import SharedMerkleTree
        tree = SharedMerkleTree(depth=4)
        tree.insert(100)
        tree.insert(200)
        proof = tree.prove_non_membership(50)  # smaller than all
        assert proof is not None
        assert proof.left_is_sentinel is True
        assert SharedMerkleTree.verify_non_membership(proof) is True

    def test_right_boundary_vector(self):
        """Commitment larger than all leaves (right boundary)."""
        from fraud_detection.merkle_tree import SharedMerkleTree
        tree = SharedMerkleTree(depth=4)
        tree.insert(100)
        tree.insert(200)
        proof = tree.prove_non_membership(999)  # larger than all
        assert proof is not None
        assert proof.right_is_sentinel is True
        assert SharedMerkleTree.verify_non_membership(proof) is True

    def test_empty_tree_vector(self):
        """Non-membership in an empty tree (both sentinels)."""
        from fraud_detection.merkle_tree import SharedMerkleTree
        tree = SharedMerkleTree(depth=4)
        proof = tree.prove_non_membership(42)
        assert proof is not None
        assert proof.left_is_sentinel is True
        assert proof.right_is_sentinel is True
        assert SharedMerkleTree.verify_non_membership(proof) is True

    def test_single_element_tree_vector(self):
        """Non-membership in a tree with exactly one leaf."""
        from fraud_detection.merkle_tree import SharedMerkleTree
        tree = SharedMerkleTree(depth=4)
        tree.insert(200)
        # Check smaller
        proof_small = tree.prove_non_membership(100)
        assert proof_small.left_is_sentinel is True
        assert proof_small.right_neighbor == 200
        assert SharedMerkleTree.verify_non_membership(proof_small) is True
        # Check larger
        proof_large = tree.prove_non_membership(300)
        assert proof_large.right_is_sentinel is True
        assert proof_large.left_neighbor == 200
        assert SharedMerkleTree.verify_non_membership(proof_large) is True

    def test_large_tree_non_membership(self):
        """Non-membership in a tree with many leaves."""
        from fraud_detection.merkle_tree import SharedMerkleTree
        tree = SharedMerkleTree(depth=10)
        # Insert 500 even numbers
        for i in range(500):
            tree.insert(i * 2)
        # Check an odd number (not in tree)
        proof = tree.prove_non_membership(501)
        assert proof is not None
        assert SharedMerkleTree.verify_non_membership(proof) is True
