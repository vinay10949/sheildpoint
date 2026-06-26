"""
ShieldPoint Cross-Party ZKP Fraud Detection Network (SP-500 / SP-501)
=====================================================================

This package implements the three components of the cross-party fraud
detection network:

1. **Commitment Generation Service** (:mod:`commitment`)
   Creates Poseidon hashes of ``claimant_id || incident_hash || salt``
   for claim deduplication across insurers. Only the commitment is
   shared — never raw claim data.

2. **Shared Merkle Tree** (:mod:`merkle_tree`)
   A sorted Merkle tree of commitments maintained by the coordination
   layer. Supports incremental insertion and Merkle proof generation
   (both membership and non-membership).

3. **Non-Membership Prover** (:mod:`non_membership_prover`)
   Python wrapper around the Circom non-membership circuit. Auto-detects
   compiled Groth16 artifacts; falls back to a deterministic stub prover
   when artifacts are absent (for local development and CI).

Public API
----------
    from fraud_detection import (
        CommitmentService,
        SharedMerkleTree,
        NonMembershipProver,
        FraudDetectionClient,
    )
"""

from .commitment import CommitmentService, generate_commitment
from .merkle_tree import SharedMerkleTree, MerkleProof, NonMembershipProof
from .non_membership_prover import (
    NonMembershipProver,
    NonMembershipResult,
    ProverMode,
)
from .client import FraudDetectionClient, FraudDetectionResult

__all__ = [
    "CommitmentService",
    "generate_commitment",
    "SharedMerkleTree",
    "MerkleProof",
    "NonMembershipProof",
    "NonMembershipProver",
    "NonMembershipResult",
    "ProverMode",
    "FraudDetectionClient",
    "FraudDetectionResult",
]
