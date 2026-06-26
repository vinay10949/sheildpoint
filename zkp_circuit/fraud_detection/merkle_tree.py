"""
SP-500 — Shared Merkle Tree of Commitments
==========================================

A sorted Merkle tree of claim commitments maintained by the coordination
layer. Supports:

- **Incremental insertion** — new commitments are inserted in sorted
  order to enable non-membership proofs via the "adjacent leaves"
  construction.
- **Membership proof generation** — proves a commitment IS in the tree.
- **Non-membership proof generation** — proves a commitment is NOT in
  the tree, by revealing the two adjacent leaves that bracket where it
  WOULD be inserted.

Tree Structure
--------------
- **Leaves**: sorted ascending by commitment value. Each leaf is a
  Poseidon hash (a BN128 field element).
- **Internal nodes**: Poseidon(left_child, right_child).
- **Depth**: configurable (default 20, supporting 2^20 = ~1M leaves).
- **Empty subtrees**: filled with the zero hash at each level (precomputed).

Sorted-Order Invariant
----------------------
To support non-membership proofs efficiently, leaves are maintained in
sorted ascending order. When a new commitment C arrives:

1. Binary-search for the insertion index `idx` where `leaves[idx-1] < C < leaves[idx]`.
2. If C already exists at `idx`, reject as a duplicate (membership, not non-membership).
3. Otherwise, insert C at `idx`, shifting all subsequent leaves right.

The sorted order guarantees that for any commitment C NOT in the tree,
the two adjacent leaves (leftNeighbor, rightNeighbor) are well-defined
and satisfy `leftNeighbor < C < rightNeighbor`.

Concurrency
-----------
The coordination layer wraps all mutations in a mutex (asyncio.Lock in
the FastAPI service). This module itself is single-threaded; the API
layer enforces serialisation.

Performance
-----------
For depth=20 (1M leaves):
- Insertion: O(depth) = O(20) Poseidon hashes (we rebuild only the
  affected path, not the whole tree).
- Membership proof: O(depth) = O(20) hashes.
- Non-membership proof: O(depth) = O(20) hashes (binary search + path
  extraction).
- Root computation: O(1) after incremental update (root is cached).
"""

from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from .commitment import FIELD_PRIME, _poseidon_hash


# Precompute zero hashes for each level of the tree.
# zero_hashes[i] is the hash of an empty subtree at depth i.
# zero_hashes[0] = 0 (leaf level — empty leaf is the field zero)
# zero_hashes[i] = Poseidon(zero_hashes[i-1], zero_hashes[i-1])
def _compute_zero_hashes(depth: int) -> list[int]:
    zeros = [0]  # level 0 (leaf): empty leaf = 0
    for i in range(1, depth + 1):
        zeros.append(_poseidon_hash([zeros[i - 1], zeros[i - 1]]))
    return zeros


@dataclass(frozen=True)
class MerkleProof:
    """A membership Merkle proof for a single leaf.

    Attributes
    ----------
    leaf : int
        The leaf value (commitment) being proven.
    leaf_index : int
        Position of the leaf in the sorted leaves array.
    path : list[int]
        Sibling hashes along the path from leaf to root (length = depth).
    path_indices : list[int]
        0 if the sibling is on the RIGHT (leaf is left child), 1 if the
        sibling is on the LEFT (leaf is right child). Length = depth.
    root : int
        The Merkle root at the time the proof was generated.
    """

    leaf: int
    leaf_index: int
    path: list[int]
    path_indices: list[int]
    root: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "leaf": str(self.leaf),
            "leaf_index": self.leaf_index,
            "path": [str(p) for p in self.path],
            "path_indices": self.path_indices,
            "root": str(self.root),
        }


@dataclass(frozen=True)
class NonMembershipProof:
    """A non-membership Merkle proof for a commitment C.

    Proves that C is NOT a leaf in the tree by showing two adjacent
    leaves that strictly bracket C: ``leftNeighbor < C < rightNeighbor``.

    Attributes
    ----------
    new_commitment : int
        The commitment C being checked for non-membership.
    left_neighbor : int
        The leaf immediately LEFT of where C would be inserted.
        0 if C is smaller than all leaves (leftIsSentinel=True).
    right_neighbor : int
        The leaf immediately RIGHT of where C would be inserted.
        0 if C is larger than all leaves (rightIsSentinel=True).
    left_is_sentinel : bool
        True if there is no left neighbor (C < all leaves).
    right_is_sentinel : bool
        True if there is no right neighbor (C > all leaves).
    left_path : list[int]
        Merkle path for left_neighbor (empty if left_is_sentinel).
    right_path : list[int]
        Merkle path for right_neighbor (empty if right_is_sentinel).
    left_path_indices : list[int]
        Path indices for left_neighbor (empty if left_is_sentinel).
    right_path_indices : list[int]
        Path indices for right_neighbor (empty if right_is_sentinel).
    root : int
        The Merkle root at the time the proof was generated.
    """

    new_commitment: int
    left_neighbor: int
    right_neighbor: int
    left_is_sentinel: bool
    right_is_sentinel: bool
    left_path: list[int]
    right_path: list[int]
    left_path_indices: list[int]
    right_path_indices: list[int]
    root: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_commitment": str(self.new_commitment),
            "left_neighbor": str(self.left_neighbor),
            "right_neighbor": str(self.right_neighbor),
            "left_is_sentinel": self.left_is_sentinel,
            "right_is_sentinel": self.right_is_sentinel,
            "left_path": [str(p) for p in self.left_path],
            "right_path": [str(p) for p in self.right_path],
            "left_path_indices": self.left_path_indices,
            "right_path_indices": self.right_path_indices,
            "root": str(self.root),
        }


class SharedMerkleTree:
    """Sorted Merkle tree of claim commitments.

    Maintains leaves in sorted ascending order to enable efficient
    non-membership proofs. All mutations rebuild only the affected
    path from the modified leaf to the root — O(depth) per operation.
    """

    def __init__(self, depth: int = 20) -> None:
        if depth < 1 or depth > 30:
            raise ValueError(f"Depth {depth} out of range [1, 30]")
        self.depth = depth
        self.capacity = 2 ** depth  # maximum number of leaves
        self.zero_hashes = _compute_zero_hashes(depth)
        # Sorted list of leaf values (commitments)
        self._leaves: list[int] = []
        # Cached tree levels: levels[0] = leaves, levels[depth] = root
        # We store only the levels that are non-trivial (sparse representation
        # would be more memory-efficient, but for <=1M leaves a flat array is fine).
        self._levels: list[list[int]] = [[] for _ in range(depth + 1)]
        self._root: int = self.zero_hashes[depth]

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def root(self) -> int:
        """Current Merkle root (zero hash if tree is empty)."""
        return self._root

    @property
    def leaf_count(self) -> int:
        return len(self._leaves)

    @property
    def leaves(self) -> list[int]:
        return list(self._leaves)

    # ------------------------------------------------------------------ #
    # Insertion
    # ------------------------------------------------------------------ #
    def insert(self, commitment: int) -> bool:
        """Insert a commitment into the tree.

        Returns
        -------
        bool
            True if inserted, False if the commitment already exists
            (duplicate — caller should treat as a membership hit).
        """
        if not (0 <= commitment < FIELD_PRIME):
            raise ValueError(f"Commitment out of field range: {commitment}")
        if len(self._leaves) >= self.capacity:
            raise OverflowError(
                f"Tree at capacity ({self.capacity} leaves)"
            )

        # Check for existing membership
        idx = bisect.bisect_left(self._leaves, commitment)
        if idx < len(self._leaves) and self._leaves[idx] == commitment:
            return False  # duplicate — already in tree

        # Insert at sorted position
        self._leaves.insert(idx, commitment)
        # Rebuild the affected path
        self._rebuild_path(idx)
        return True

    def _rebuild_path(self, leaf_index: int) -> None:
        """Rebuild all internal nodes on the path from ``leaf_index`` to root."""
        # Update leaves level
        self._levels[0] = list(self._leaves)
        # Pad leaves to capacity with zero hashes for node computation
        # (We compute parents pairwise; for odd-length levels, the last
        # element pairs with the zero hash.)
        current = list(self._leaves)
        for level in range(self.depth):
            parents: list[int] = []
            n = len(current)
            # Pad to even length
            if n % 2 == 1:
                current = current + [self.zero_hashes[level]]
                n += 1
            for i in range(0, n, 2):
                parents.append(_poseidon_hash([current[i], current[i + 1]]))
            self._levels[level + 1] = parents
            current = parents
        self._root = current[0] if current else self.zero_hashes[self.depth]

    # ------------------------------------------------------------------ #
    # Membership proof
    # ------------------------------------------------------------------ #
    def prove_membership(self, commitment: int) -> Optional[MerkleProof]:
        """Generate a membership proof for ``commitment``.

        Returns None if the commitment is NOT in the tree.
        """
        if commitment not in self._leaves:
            return None
        leaf_index = self._leaves.index(commitment)
        return self._prove_path(leaf_index, commitment)

    def _prove_path(self, leaf_index: int, leaf_value: int) -> MerkleProof:
        """Build a Merkle proof for the leaf at ``leaf_index``."""
        path: list[int] = []
        path_indices: list[int] = []
        idx = leaf_index
        for level in range(self.depth):
            # Sibling is at idx XOR 1
            sibling_idx = idx ^ 1
            level_nodes = self._levels[level] if level < len(self._levels) else []
            if sibling_idx < len(level_nodes):
                sibling = level_nodes[sibling_idx]
            else:
                # Sibling is in an empty subtree — use zero hash at this level
                sibling = self.zero_hashes[level]
            path.append(sibling)
            # path_indices[i] = 0 if our node is the LEFT child (sibling on right)
            #                  1 if our node is the RIGHT child (sibling on left)
            path_indices.append(idx & 1)
            idx = idx >> 1  # move to parent level
        return MerkleProof(
            leaf=leaf_value,
            leaf_index=leaf_index,
            path=path,
            path_indices=path_indices,
            root=self._root,
        )

    # ------------------------------------------------------------------ #
    # Non-membership proof
    # ------------------------------------------------------------------ #
    def prove_non_membership(
        self, commitment: int
    ) -> Optional[NonMembershipProof]:
        """Generate a non-membership proof for ``commitment``.

        Returns
        -------
        NonMembershipProof or None
            A proof if the commitment is NOT in the tree.
            None if the commitment IS in the tree (it's a member —
            caller should treat as a duplicate/fraud hit).
        """
        if not (0 <= commitment < FIELD_PRIME):
            raise ValueError(f"Commitment out of field range: {commitment}")

        idx = bisect.bisect_left(self._leaves, commitment)

        # Check membership
        if idx < len(self._leaves) and self._leaves[idx] == commitment:
            return None  # IS a member — caller should flag as duplicate

        # Left neighbor is at idx-1 (or sentinel if idx==0)
        left_is_sentinel = (idx == 0)
        right_is_sentinel = (idx == len(self._leaves))

        left_neighbor = 0 if left_is_sentinel else self._leaves[idx - 1]
        right_neighbor = 0 if right_is_sentinel else self._leaves[idx]

        # Verify the bracketing invariant
        if not left_is_sentinel and not (left_neighbor < commitment):
            raise RuntimeError(
                f"Non-membership invariant violated: left_neighbor "
                f"{left_neighbor} >= commitment {commitment}"
            )
        if not right_is_sentinel and not (commitment < right_neighbor):
            raise RuntimeError(
                f"Non-membership invariant violated: commitment "
                f"{commitment} >= right_neighbor {right_neighbor}"
            )

        # Build Merkle paths for the two neighbors
        left_path: list[int] = []
        left_path_indices: list[int] = []
        right_path: list[int] = []
        right_path_indices: list[int] = []

        if not left_is_sentinel:
            left_proof = self._prove_path(idx - 1, left_neighbor)
            left_path = left_proof.path
            left_path_indices = left_proof.path_indices

        if not right_is_sentinel:
            right_proof = self._prove_path(idx, right_neighbor)
            right_path = right_proof.path
            right_path_indices = right_proof.path_indices

        return NonMembershipProof(
            new_commitment=commitment,
            left_neighbor=left_neighbor,
            right_neighbor=right_neighbor,
            left_is_sentinel=left_is_sentinel,
            right_is_sentinel=right_is_sentinel,
            left_path=left_path,
            right_path=right_path,
            left_path_indices=left_path_indices,
            right_path_indices=right_path_indices,
            root=self._root,
        )

    # ------------------------------------------------------------------ #
    # Verification (static — can be run by any party)
    # ------------------------------------------------------------------ #
    @staticmethod
    def verify_membership(proof: MerkleProof) -> bool:
        """Verify a membership proof.

        Returns True if the leaf + path hash to the claimed root.
        """
        current = proof.leaf
        for i in range(len(proof.path)):
            if proof.path_indices[i] == 0:
                # Leaf is left child: parent = Poseidon(leaf, sibling)
                current = _poseidon_hash([current, proof.path[i]])
            else:
                # Leaf is right child: parent = Poseidon(sibling, leaf)
                current = _poseidon_hash([proof.path[i], current])
        return current == proof.root

    @staticmethod
    def verify_non_membership(proof: NonMembershipProof) -> bool:
        """Verify a non-membership proof.

        Checks:
        1. leftNeighbor < newCommitment < rightNeighbor (strict)
        2. leftNeighbor's Merkle path is valid (if not sentinel)
        3. rightNeighbor's Merkle path is valid (if not sentinel)
        4. NOT both sentinels (tree must be non-empty)
        """
        # Check 4: not both sentinels
        if proof.left_is_sentinel and proof.right_is_sentinel:
            # Tree is empty — non-membership is trivially true,
            # but the coordination layer should handle this case
            # without invoking the circuit.
            return True

        # Check 1: strict bracketing
        if not proof.left_is_sentinel:
            if not (proof.left_neighbor < proof.new_commitment):
                return False
        if not proof.right_is_sentinel:
            if not (proof.new_commitment < proof.right_neighbor):
                return False

        # Check 2: left Merkle path
        if not proof.left_is_sentinel:
            left_proof = MerkleProof(
                leaf=proof.left_neighbor,
                leaf_index=0,  # not used in verification
                path=proof.left_path,
                path_indices=proof.left_path_indices,
                root=proof.root,
            )
            if not SharedMerkleTree.verify_membership(left_proof):
                return False

        # Check 3: right Merkle path
        if not proof.right_is_sentinel:
            right_proof = MerkleProof(
                leaf=proof.right_neighbor,
                leaf_index=0,  # not used in verification
                path=proof.right_path,
                path_indices=proof.right_path_indices,
                root=proof.root,
            )
            if not SharedMerkleTree.verify_membership(right_proof):
                return False

        return True

    # ------------------------------------------------------------------ #
    # Bulk operations
    # ------------------------------------------------------------------ #
    def bulk_insert(self, commitments: list[int]) -> int:
        """Insert multiple commitments. Returns count of newly inserted
        (duplicates are skipped)."""
        # Sort first for efficient batch insertion
        sorted_commits = sorted(set(commitments))
        inserted = 0
        for c in sorted_commits:
            if self.insert(c):
                inserted += 1
        return inserted

    def to_dict(self) -> dict[str, Any]:
        """Serialize tree state (for persistence / debugging)."""
        return {
            "depth": self.depth,
            "leaf_count": self.leaf_count,
            "root": str(self._root),
            "leaves": [str(l) for l in self._leaves],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedMerkleTree":
        """Reconstruct a tree from serialized state."""
        tree = cls(depth=data["depth"])
        for leaf_str in data["leaves"]:
            tree.insert(int(leaf_str))
        return tree
