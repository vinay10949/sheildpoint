/*
 * ShieldPoint Policy Validity Proof Circuit — Extended (v2)
 * =========================================================
 *
 * This zk-SNARK circuit proves that an insurance claim is valid under a given
 * policy without revealing the underlying policy or claim details. The circuit
 * verifies:
 *
 *   1. The policy commitment corresponds to a valid active policy
 *      (Poseidon(policyId, salt) == policyCommitment)
 *   2. The claimed peril is within covered perils (membership check)
 *   3. The claimed peril is NOT in excluded perils (anti-membership check)
 *   4. The date of loss falls within the effective period
 *   5. The claim amount does not exceed the coverage limit
 *   6. The net claim (amount - deductible) does not exceed the
 *      net coverage limit (coverageLimit - deductible)
 *
 * v2 additions over v1:
 *   - Exclusion check: perilType must NOT be in exclusions[8]
 *   - Deductible application: (claimAmount - deductible) <= (coverageLimit - deductible)
 *     which simplifies to claimAmount <= coverageLimit (always true if original
 *     limit check passes). However, we add an explicit net-amount check:
 *     claimAmount >= deductible (claim must exceed deductible to be valid)
 *
 * Public Inputs (known to verifier):
 *   - policyCommitment: Poseidon hash of (policyId, salt)
 *   - claimType:        Numeric peril identifier
 *
 * Private Inputs (known only to prover):
 *   - policyId, salt:                For commitment verification
 *   - coverageLimit, deductible:     Policy financial parameters
 *   - effectiveDate, expirationDate: Policy date range (days since epoch)
 *   - perils[8]:                     Covered peril type codes (0 = unused slot)
 *   - exclusions[8]:                 Excluded peril type codes (0 = unused slot)
 *   - policyStatus:                  1 = active, 0 = inactive
 *   - claimAmount:                   Amount being claimed
 *   - perilType:                     Peril code of the claim event
 *   - dateOfLoss:                    Date of the loss event (days since epoch)
 *
 * Output:
 *   - isValid: 1 if all conditions hold, 0 otherwise
 */

pragma circom 2.1.9;

include "../node_modules/circomlib/circuits/poseidon.circom";
include "../node_modules/circomlib/circuits/comparators.circom";

/* ========================================================================
 * RangeCheck: verifies that effectiveDate <= dateOfLoss <= expirationDate
 * Returns 1 if the date is within range, 0 otherwise.
 * ======================================================================== */
template RangeCheck(n) {
    signal input effectiveDate;
    signal input dateOfLoss;
    signal input expirationDate;
    signal output out;

    component leq_lower = LessEqThan(n);
    leq_lower.in[0] <== effectiveDate;
    leq_lower.in[1] <== dateOfLoss;

    component leq_upper = LessEqThan(n);
    leq_upper.in[0] <== dateOfLoss;
    leq_upper.in[1] <== expirationDate;

    out <== leq_lower.out * leq_upper.out;
}

/* ========================================================================
 * PerilMembership: verifies that perilType is in the list of covered perils
 * Returns 1 if peril is covered, 0 otherwise.
 * ======================================================================== */
template PerilMembership(n_perils) {
    signal input perilType;
    signal input perils[n_perils];
    signal output out;

    signal isMatch[n_perils];

    component eq[n_perils];
    for (var i = 0; i < n_perils; i++) {
        eq[i] = IsEqual();
        eq[i].in[0] <== perils[i];
        eq[i].in[1] <== perilType;
        isMatch[i] <== eq[i].out;
    }

    // OR all matches: out = 1 - prod(1 - isMatch[i])
    signal accum[n_perils + 1];
    accum[0] <== 1;
    for (var i = 0; i < n_perils; i++) {
        accum[i + 1] <== accum[i] * (1 - isMatch[i]);
    }
    out <== 1 - accum[n_perils];
}

/* ========================================================================
 * PerilExclusion: verifies that perilType is NOT in the exclusion list
 * Returns 1 if peril is NOT excluded, 0 if it IS excluded.
 * ======================================================================== */
template PerilExclusion(n_exclusions) {
    signal input perilType;
    signal input exclusions[n_exclusions];
    signal output out;

    signal isExcluded[n_exclusions];

    component eq[n_exclusions];
    for (var i = 0; i < n_exclusions; i++) {
        eq[i] = IsEqual();
        eq[i].in[0] <== exclusions[i];
        eq[i].in[1] <== perilType;
        isExcluded[i] <== eq[i].out;
    }

    // OR all exclusions: excluded = 1 - prod(1 - isExcluded[i])
    signal accum[n_exclusions + 1];
    accum[0] <== 1;
    for (var i = 0; i < n_exclusions; i++) {
        accum[i + 1] <== accum[i] * (1 - isExcluded[i]);
    }
    signal excluded;
    excluded <== 1 - accum[n_exclusions];

    // NOT excluded: out = 1 - excluded
    out <== 1 - excluded;
}

/* ========================================================================
 * DeductibleCheck: verifies that claimAmount >= deductible
 * The claim must exceed the deductible to be valid.
 * Returns 1 if claimAmount >= deductible, 0 otherwise.
 * ======================================================================== */
template DeductibleCheck(n) {
    signal input claimAmount;
    signal input deductible;
    signal output out;

    component geq = GreaterEqThan(n);
    geq.in[0] <== claimAmount;
    geq.in[1] <== deductible;

    out <== geq.out;
}

/* ========================================================================
 * PolicyValidityProof — Main Circuit (v2)
 * ======================================================================== */
template PolicyValidityProof() {
    // === Public Inputs ===
    signal input policyCommitment;
    signal input claimType;

    // === Private Inputs: Policy ===
    signal input policyId;
    signal input salt;
    signal input coverageLimit;
    signal input deductible;
    signal input effectiveDate;
    signal input expirationDate;
    signal input perils[8];
    signal input exclusions[8];       // NEW: Excluded peril codes
    signal input policyStatus;

    // === Private Inputs: Claim ===
    signal input claimAmount;
    signal input perilType;
    signal input dateOfLoss;

    // === Public Output ===
    signal output isValid;

    // ------------------------------------------------------------------
    // Constraint 1: Policy Commitment Verification
    // ------------------------------------------------------------------
    component poseidon = Poseidon(2);
    poseidon.inputs[0] <== policyId;
    poseidon.inputs[1] <== salt;

    component commitment_eq = IsEqual();
    commitment_eq.in[0] <== poseidon.out;
    commitment_eq.in[1] <== policyCommitment;
    signal commitmentValid;
    commitmentValid <== commitment_eq.out;

    // ------------------------------------------------------------------
    // Constraint 2: Policy Status is Active
    // ------------------------------------------------------------------
    signal statusSquared;
    statusSquared <== policyStatus * policyStatus;
    statusSquared === policyStatus;

    // ------------------------------------------------------------------
    // Constraint 3: Peril Coverage Check (membership)
    // ------------------------------------------------------------------
    component perilCheck = PerilMembership(8);
    perilCheck.perilType <== perilType;
    for (var i = 0; i < 8; i++) {
        perilCheck.perils[i] <== perils[i];
    }
    signal perilCovered;
    perilCovered <== perilCheck.out;

    // Enforce: perilType === claimType
    perilType === claimType;

    // ------------------------------------------------------------------
    // Constraint 4: Peril Exclusion Check (anti-membership) [NEW in v2]
    // The peril must NOT be in the exclusion list.
    // ------------------------------------------------------------------
    component exclusionCheck = PerilExclusion(8);
    exclusionCheck.perilType <== perilType;
    for (var i = 0; i < 8; i++) {
        exclusionCheck.exclusions[i] <== exclusions[i];
    }
    signal perilNotExcluded;
    perilNotExcluded <== exclusionCheck.out;

    // ------------------------------------------------------------------
    // Constraint 5: Date Range Validation
    // ------------------------------------------------------------------
    component dateCheck = RangeCheck(64);
    dateCheck.effectiveDate <== effectiveDate;
    dateCheck.dateOfLoss <== dateOfLoss;
    dateCheck.expirationDate <== expirationDate;
    signal dateInRange;
    dateInRange <== dateCheck.out;

    // ------------------------------------------------------------------
    // Constraint 6: Coverage Limit Check
    // claimAmount <= coverageLimit
    // ------------------------------------------------------------------
    component limitCheck = LessEqThan(64);
    limitCheck.in[0] <== claimAmount;
    limitCheck.in[1] <== coverageLimit;
    signal withinLimit;
    withinLimit <== limitCheck.out;

    // ------------------------------------------------------------------
    // Constraint 7: Deductible Application [NEW in v2]
    // claimAmount >= deductible (claim must exceed deductible)
    // ------------------------------------------------------------------
    component deductibleCheck = DeductibleCheck(64);
    deductibleCheck.claimAmount <== claimAmount;
    deductibleCheck.deductible <== deductible;
    signal exceedsDeductible;
    exceedsDeductible <== deductibleCheck.out;

    // ------------------------------------------------------------------
    // Combine all constraints: AND of all checks (pairwise)
    // ------------------------------------------------------------------
    signal and1;
    and1 <== commitmentValid * policyStatus;

    signal and2;
    and2 <== and1 * perilCovered;

    signal and3;
    and3 <== and2 * perilNotExcluded;

    signal and4;
    and4 <== and3 * dateInRange;

    signal and5;
    and5 <== and4 * withinLimit;

    isValid <== and5 * exceedsDeductible;
}

// Top-level: policyCommitment and claimType are public
component main {public [policyCommitment, claimType]} = PolicyValidityProof();
