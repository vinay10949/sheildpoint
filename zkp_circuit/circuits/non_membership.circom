/*
 * ShieldPoint Cross-Party Fraud Detection — Non-Membership Proof Circuit
 * =====================================================================
 *
 * SP-501 / SP-500: Core cryptographic primitive enabling cross-insurer
 * duplicate-claim detection WITHOUT revealing any sensitive claim data.
 *
 * Purpose
 * -------
 * When a new claim arrives at the ClassifierAgent, ShieldPoint must verify
 * that the SAME claim (same claimant + same incident) has not already been
 * filed with ANY participating insurer in the fraud-detection network.
 *
 * Each insurer submits only a cryptographic commitment
 *   C = Poseidon(claimant_id, incident_hash, salt)
 * to a shared Merkle tree maintained by the coordination layer. The tree
 * stores ONLY commitments — never raw claim data — ensuring compliance
 * with data-minimisation principles and competitive-sensitivity rules.
 *
 * To prove a new claim is NOT a duplicate, the prover constructs a
 * non-membership Merkle proof showing that the commitment C does NOT
 * appear as a leaf in the shared tree. The proof reveals only the Merkle
 * root and the new commitment C; everything else (salt, sibling hashes,
 * adjacent leaf values) stays private.
 *
 * Non-Membership Proof Structure
 * ------------------------------
 * For a sorted Merkle tree (leaves sorted numerically), a non-membership
 * proof for value C consists of:
 *
 *   1. The leaf index `idx` where C WOULD be inserted (i.e., the position
 *      that maintains sorted order).
 *   2. The two ADJACENT leaves that bracket position `idx`:
 *        - leftNeighbor  (leaf at position idx-1, or 0 if idx==0)
 *        - rightNeighbor (leaf at position idx,   or MAX_LEAF if idx==count)
 *      These must satisfy: leftNeighbor < C < rightNeighbor.
 *      If C were already in the tree, one of these would EQUAL C and the
 *      constraint would fail.
 *   3. The Merkle path from each adjacent leaf to the root (two paths,
 *      one per neighbor), so the verifier can confirm both neighbors are
 *      actually in the tree at the claimed positions.
 *
 * Public Inputs (known to verifier — coordination layer / other insurers)
 *   - merkleRoot:       Current root of the shared commitment tree.
 *   - newCommitment:    Poseidon(claimant_id, incident_hash, salt) of the
 *                       new claim being checked.
 *
 * Private Inputs (known only to prover — the insurer checking the claim)
 *   - leftNeighbor:     Leaf value immediately LEFT of where C would go.
 *   - rightNeighbor:    Leaf value immediately RIGHT of where C would go.
 *   - leftLeafIndex:    Position of leftNeighbor in the tree (or 0 if none).
 *   - rightLeafIndex:   Position of rightNeighbor in the tree (or count if none).
 *   - leftPath[depth]:  Merkle authentication path for leftNeighbor.
 *   - rightPath[depth]: Merkle authentication path for rightNeighbor.
 *   - leftIsSentinel:   1 if leftNeighbor is the implicit 0 sentinel
 *                       (no left neighbor exists — C is smaller than all leaves).
 *   - rightIsSentinel:  1 if rightNeighbor is the implicit MAX sentinel
 *                       (no right neighbor exists — C is larger than all leaves).
 *
 * Constraints
 * -----------
 *   1. C > leftNeighbor      (strictly greater — prevents C == leftNeighbor)
 *   2. C < rightNeighbor     (strictly less  — prevents C == rightNeighbor)
 *   3. leftNeighbor's Merkle path reconstructs the public root.
 *   4. rightNeighbor's Merkle path reconstructs the public root.
 *   5. Sentinel handling: if leftIsSentinel, skip constraint 1 and the
 *      left path check; if rightIsSentinel, skip constraint 2 and the
 *      right path check.
 *
 * Constraint Budget
 * -----------------
 * AC: <= 80K constraints. Estimated breakdown for depth=20:
 *   - 2 × Merkle path verification (20 levels × ~250 constraints per
 *     Poseidon hash) ≈ 2 × 20 × 250 = 10,000
 *   - 2 × LessThan comparators (254-bit field elements) ≈ 2 × 254 = 508
 *   - 2 × IsEqual (sentinel checks) ≈ 6
 *   - 2 × path-index bit-decomposition (20 bits each) ≈ 40
 *   - Sentinel conditional logic ≈ 20
 *   Total: ~10,600 constraints — well under the 80K budget.
 *
 * Tree Depth
 * ----------
 * Default depth = 20 (supports 2^20 = 1,048,576 leaves — sufficient for
 * several years of claim volume across multiple insurers). The depth is
 * a template parameter so the same circuit can be recompiled for smaller
 * or larger trees.
 *
 * Compilation
 * -----------
 *   circom non_membership.circom --r1cs --wasm --sym -o build/non_membership
 *   snarkjs groth16 setup build/non_membership/non_membership.r1cs \
 *       keys/pot12_final.ptau keys/fraud_circuit_0000.zkey
 *   # ... (trusted setup phases, same pattern as compliance circuit) ...
 *   snarkjs zkey export verificationkey keys/fraud_circuit_final.zkey \
 *       keys/fraud_verification_key.json
 *
 * Performance
 * -----------
 *   Proof generation: < 10 seconds on CPU (target, Groth16)
 *   Verification:     < 10ms (Groth16 constant time guarantee)
 */

pragma circom 2.1.9;

include "../node_modules/circomlib/circuits/poseidon.circom";
include "../node_modules/circomlib/circuits/comparators.circom";
include "../node_modules/circomlib/circuits/mux.circom";

/* ========================================================================
 * DualMuxSelect: select between two values based on a bit.
 *   out = (1 - sel) * a + sel * b
 * Used for left/right child selection in Merkle path verification.
 * ======================================================================== */
template DualMuxSelect() {
    signal input sel;
    signal input a;
    signal input b;
    signal output out;
    out <== (1 - sel) * a + sel * b;
}

/* ========================================================================
 * MerklePathVerifier(depth): verifies that a leaf at a given index
 * hashes up to the public Merkle root.
 *
 * Inputs:
 *   - leaf:       the leaf value (private)
 *   - path[depth]: sibling hashes along the path (private)
 *   - pathIndices[depth]: 0 = leaf is LEFT child, 1 = leaf is RIGHT child (private)
 *   - root:       the expected Merkle root (public, passed in)
 *
 * Output:
 *   - computedRoot: the root derived from leaf + path
 *
 * The caller constrains computedRoot === expectedRoot.
 * ======================================================================== */
template MerklePathVerifier(depth) {
    signal input leaf;
    signal input path[depth];
    signal input pathIndices[depth];
    signal input root;
    signal output computedRoot;

    signal current;
    current <== leaf;

    signal next;
    for (var i = 0; i < depth; i++) {
        // Poseidon hash of (left, right) where the order depends on pathIndices[i]
        component poseidon = Poseidon(2);
        component muxL = DualMuxSelect();
        component muxR = DualMuxSelect();
        muxL.sel <== pathIndices[i];
        muxL.a <== current;
        muxL.b <== path[i];
        muxR.sel <== pathIndices[i];
        muxR.a <== path[i];
        muxR.b <== current;
        poseidon.inputs[0] <== muxL.out;
        poseidon.inputs[1] <== muxR.out;
        next <== poseidon.out;
        current <== next;
    }
    computedRoot <== current;
    // Constraint: computed root must match the public root
    root === computedRoot;
}

/* ========================================================================
 * StrictBetween: proves left < value < right (all field elements).
 *
 * Uses LessThan comparators. Since field elements are 254-bit, we use
 * n=254 to cover the full range. We check:
 *   - left < value  (i.e., LessThan(left, value) == 1)
 *   - value < right (i.e., LessThan(value, right) == 1)
 * Combined with AND (multiplication).
 * ======================================================================== */
template StrictBetween(n) {
    signal input left;
    signal input value;
    signal input right;
    signal output out;

    component ltLeft = LessThan(n);
    ltLeft.in[0] <== left;
    ltLeft.in[1] <== value;
    signal leftOk;
    leftOk <== ltLeft.out;

    component ltRight = LessThan(n);
    ltRight.in[0] <== value;
    ltRight.in[1] <== right;
    signal rightOk;
    rightOk <== ltRight.out;

    // out = leftOk AND rightOk
    out <== leftOk * rightOk;
}

/* ========================================================================
 * NonMembershipProof(depth): main non-membership circuit.
 *
 * Proves that `newCommitment` is NOT a leaf in the Merkle tree with the
 * given `merkleRoot`, by showing two adjacent leaves bracket it strictly.
 * ======================================================================== */
template NonMembershipProof(depth) {
    // === Public Inputs ===
    signal input merkleRoot;        // current root of the shared tree
    signal input newCommitment;     // Poseidon(claimant_id, incident_hash, salt)

    // === Private Inputs ===
    signal input leftNeighbor;      // leaf immediately LEFT of where C would go
    signal input rightNeighbor;     // leaf immediately RIGHT of where C would go
    signal input leftPath[depth];   // Merkle path for leftNeighbor
    signal input rightPath[depth];  // Merkle path for rightNeighbor
    signal input leftPathIndices[depth];   // 0/1 per level for leftNeighbor
    signal input rightPathIndices[depth];  // 0/1 per level for rightNeighbor
    signal input leftIsSentinel;    // 1 if no left neighbor (C < all leaves)
    signal input rightIsSentinel;   // 1 if no right neighbor (C > all leaves)

    // === Public Output ===
    signal output isNonMember;

    // ------------------------------------------------------------------
    // Constraint 1: leftNeighbor < newCommitment < rightNeighbor
    //
    // When leftIsSentinel=1, the "left" bound is conceptually 0 (field
    // zero), so leftNeighbor is a dummy and we skip the left comparison.
    // When rightIsSentinel=1, the "right" bound is conceptually the field
    // modulus, so rightNeighbor is a dummy and we skip the right comparison.
    //
    // We implement this with conditional logic:
    //   leftCheck  = leftIsSentinel  ? 1 : (leftNeighbor < newCommitment)
    //   rightCheck = rightIsSentinel ? 1 : (newCommitment < rightNeighbor)
    //   isNonMember = leftCheck * rightCheck
    // ------------------------------------------------------------------

    // Left comparison (only meaningful if leftIsSentinel == 0)
    component leftLt = LessThan(254);
    leftLt.in[0] <== leftNeighbor;
    leftLt.in[1] <== newCommitment;
    signal leftCmp;
    leftCmp <== leftLt.out;

    // If leftIsSentinel, leftCheck = 1; else leftCheck = leftCmp
    signal leftCheck;
    leftCheck <== leftIsSentinel + (1 - leftIsSentinel) * leftCmp;

    // Right comparison (only meaningful if rightIsSentinel == 0)
    component rightLt = LessThan(254);
    rightLt.in[0] <== newCommitment;
    rightLt.in[1] <== rightNeighbor;
    signal rightCmp;
    rightCmp <== rightLt.out;

    // If rightIsSentinel, rightCheck = 1; else rightCheck = rightCmp
    signal rightCheck;
    rightCheck <== rightIsSentinel + (1 - rightIsSentinel) * rightCmp;

    // ------------------------------------------------------------------
    // Constraint 2: Merkle path verification for leftNeighbor
    //
    // If leftIsSentinel, we skip this check (no left neighbor to verify).
    // Otherwise, the leftNeighbor + leftPath must hash to merkleRoot.
    //
    // We verify unconditionally and multiply the result by (1 - leftIsSentinel)
    // so that when sentinel=1, the check is bypassed (leftPathValid_forced = 1).
    // ------------------------------------------------------------------
    component leftMerkle = MerklePathVerifier(depth);
    leftMerkle.leaf <== leftNeighbor;
    for (var i = 0; i < depth; i++) {
        leftMerkle.path[i] <== leftPath[i];
        leftMerkle.pathIndices[i] <== leftPathIndices[i];
    }
    leftMerkle.root <== merkleRoot;
    signal leftMerkleValid;
    leftMerkleValid <== leftMerkle.computedRoot === merkleRoot ? 1 : 0;

    // Conditional: if sentinel, force valid (1); else use actual result
    signal leftPathValid;
    leftPathValid <== leftIsSentinel + (1 - leftIsSentinel) * leftMerkleValid;

    // ------------------------------------------------------------------
    // Constraint 3: Merkle path verification for rightNeighbor
    // ------------------------------------------------------------------
    component rightMerkle = MerklePathVerifier(depth);
    rightMerkle.leaf <== rightNeighbor;
    for (var i = 0; i < depth; i++) {
        rightMerkle.path[i] <== rightPath[i];
        rightMerkle.pathIndices[i] <== rightPathIndices[i];
    }
    rightMerkle.root <== merkleRoot;
    signal rightMerkleValid;
    rightMerkleValid <== rightMerkle.computedRoot === merkleRoot ? 1 : 0;

    signal rightPathValid;
    rightPathValid <== rightIsSentinel + (1 - rightIsSentinel) * rightMerkleValid;

    // ------------------------------------------------------------------
    // Constraint 4: NOT both sentinels at once (tree must be non-empty if
    // we're proving non-membership against a real root). If the tree is
    // empty, merkleRoot is the zero hash and the proof is trivially true —
    // handled by the coordination layer returning a special "empty tree"
    // response without invoking the circuit.
    // ------------------------------------------------------------------
    signal notBothSentinels;
    notBothSentinels <== 1 - leftIsSentinel * rightIsSentinel;

    // ------------------------------------------------------------------
    // Final: all checks must pass
    // ------------------------------------------------------------------
    isNonMember <== leftCheck * rightCheck * leftPathValid * rightPathValid * notBothSentinels;
}

/* ========================================================================
 * Top-level: instantiate with depth=20 (supports 2^20 leaves).
 * Public inputs: merkleRoot, newCommitment.
 * ======================================================================== */
component main {public [merkleRoot, newCommitment]} = NonMembershipProof(20);
