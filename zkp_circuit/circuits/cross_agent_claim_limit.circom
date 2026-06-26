/*
 * ShieldPoint Cross-Agent Claim Limit Proof Circuit (SP-304)
 * ==========================================================
 *
 * Purpose
 * -------
 * Allows the ClaimsAgent to prove to the FinancialAgent (and the
 * ManagerAgent at each handoff) that a claim amount is within a policy's
 * coverage limit WITHOUT revealing the underlying policy document.
 *
 * The ClaimsAgent holds the full policy (limit, deductible, perils, dates).
 * The FinancialAgent only needs to know "the claim amount is within the
 * coverage limit" to authorise payment. Today, this requires the
 * FinancialAgent to access the full policy document — an unnecessary data
 * exposure. This circuit replaces that disclosure with a zk-SNARK proof.
 *
 * Public Statement
 * ----------------
 *   "I, the ClaimsAgent, attest that for policy with commitment C,
 *    the claim amount A satisfies A <= L, where L is the coverage
 *    limit committed to by the policy."
 *
 * Public Inputs (known to verifier — FinancialAgent / ManagerAgent)
 *   - policyCommitment:   PoseidonHash(policyId, salt, coverageLimit)
 *   - claimAmount:        The amount being claimed (revealed, but only
 *                         this one number — no other policy detail)
 *
 * Private Inputs (known only to prover — ClaimsAgent)
 *   - policyId, salt:     For commitment verification
 *   - coverageLimit:      The policy's coverage limit
 *
 * Output
 *   - isValid: 1 if (a) commitment verifies AND (b) claimAmount <= coverageLimit
 *
 * Constraint Budget
 * -----------------
 * AC: < 30K constraints. This circuit uses:
 *   - 1 Poseidon(3) hash       ~  250 constraints
 *   - 1 IsEqual                ~    3 constraints
 *   - 1 LessEqThan(64)         ~  250 constraints
 *   - 1 AND (multiply)         ~    1 constraint
 *   Total: ~500 constraints — far below the 30K budget.
 *
 * Compatibility
 * -------------
 * Includes the same circomlib primitives as policy_validity.circom so the
 * existing Makefile trusted-setup pipeline (Groth16, 8 contributors) can
 * be reused with CIRCUIT_NAME=cross_agent_claim_limit.
 */

pragma circom 2.1.9;

include "../node_modules/circomlib/circuits/poseidon.circom";
include "../node_modules/circomlib/circuits/comparators.circom";

/* ========================================================================
 * CrossAgentClaimLimitProof — Main Circuit
 * ========================================================================
 */
template CrossAgentClaimLimitProof() {
    // === Public Inputs ===
    signal input policyCommitment;
    signal input claimAmount;

    // === Private Inputs ===
    signal input policyId;
    signal input salt;
    signal input coverageLimit;

    // === Public Output ===
    signal output isValid;

    // ------------------------------------------------------------------
    // Constraint 1: Policy Commitment Verification
    //   policyCommitment === Poseidon(policyId, salt, coverageLimit)
    //
    // By binding the coverageLimit into the commitment (rather than just
    // policyId+salt as in policy_validity.circom), the FinancialAgent can
    // trust that the limit used inside this proof is the SAME limit the
    // ClaimsAgent committed to publicly. This prevents a malicious prover
    // from inflating the limit at proof time.
    // ------------------------------------------------------------------
    component poseidon = Poseidon(3);
    poseidon.inputs[0] <== policyId;
    poseidon.inputs[1] <== salt;
    poseidon.inputs[2] <== coverageLimit;

    component commitment_eq = IsEqual();
    commitment_eq.in[0] <== poseidon.out;
    commitment_eq.in[1] <== policyCommitment;
    signal commitmentValid;
    commitmentValid <== commitment_eq.out;

    // ------------------------------------------------------------------
    // Constraint 2: Claim amount is within coverage limit
    //   claimAmount <= coverageLimit
    // ------------------------------------------------------------------
    component limitCheck = LessEqThan(64);
    limitCheck.in[0] <== claimAmount;
    limitCheck.in[1] <== coverageLimit;
    signal withinLimit;
    withinLimit <== limitCheck.out;

    // ------------------------------------------------------------------
    // Combine: isValid = commitmentValid AND withinLimit
    // ------------------------------------------------------------------
    isValid <== commitmentValid * withinLimit;
}

// Top-level: policyCommitment and claimAmount are public inputs.
// isValid is the public output (also exposed in public.json).
component main {public [policyCommitment, claimAmount]} = CrossAgentClaimLimitProof();
