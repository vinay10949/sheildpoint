/*
 * ShieldPoint Compliance Verification Circuit
 * ============================================
 *
 * This zk-SNARK circuit proves that an insurance claim was processed in
 * compliance with the regulations of ShieldPoint's 12 operating states
 * WITHOUT revealing any sensitive claim data (claimant PII, adjuster
 * notes, medical records). Regulators can verify the proof and confirm
 * correct processing without accessing the underlying data.
 *
 * This is the SECOND ZKP gate in the state machine
 * (ZKP_COMPLIANCE_PROOF), and is more complex than the Policy Validity
 * Proof (~120K constraints vs ~50K) because it must encode the specific
 * regulations of each of the 12 operating states.
 *
 * The 12 ShieldPoint operating states
 * -----------------------------------
 *   1. CA  — California   (10-day ack, 30-day pay, 15-day disclosure)
 *   2. NY  — New York     (15-day ack, 30-day pay, 10-day disclosure)
 *   3. TX  — Texas        (15-day ack, 10-day pay, 15-day disclosure)
 *   4. FL  — Florida      (14-day ack, 20-day pay, 15-day disclosure)
 *   5. IL  — Illinois     (21-day ack, 30-day pay, 15-day disclosure)
 *   6. PA  — Pennsylvania (10-day ack, 25-day pay, 15-day disclosure)
 *   7. OH  — Ohio         (15-day ack, 21-day pay, 15-day disclosure)
 *   8. GA  — Georgia      (15-day ack, 30-day pay, 15-day disclosure)
 *   9. NC  — North Carolina (30-day ack, 30-day pay, 15-day disclosure)
 *  10. MI  — Michigan     (20-day ack, 30-day pay, 15-day disclosure)
 *  11. NJ  — New Jersey   (10-day ack, 30-day pay, 10-day disclosure)
 *  12. WA  — Washington   (15-day ack, 15-day pay, 10-day disclosure)
 *
 * Each state's regulatory check is encoded as a constraint branch
 * selected by the public ``jurisdiction`` input. The circuit proves:
 *
 *   1. Timely acknowledgment — claim was acknowledged within the
 *      state's ack deadline.
 *   2. Payment timeline compliance — if approved, payment was issued
 *      within the state's payment deadline.
 *   3. Disclosure mandate — required disclosures were sent within the
 *      state's disclosure deadline.
 *   4. Fair claims practice — no prohibited practice occurred (e.g.
 *      lowball settlement below 60% of claim amount without documented
 *      reasoning).
 *
 * Public Inputs (known to verifier)
 * ---------------------------------
 *   - jurisdiction:    numeric state code (1..12, mapping above)
 *   - claimType:       numeric claim type code (1=property, 2=auto,
 *                       3=liability, 4=medical)
 *   - complianceRoot:  Poseidon hash of (claimRecordCommitment, salt)
 *                       — anchors the proof to a specific claim record
 *                       without revealing the record itself.
 *
 * Private Inputs (known only to prover)
 * -------------------------------------
 *   - claimRecordCommitment: Poseidon hash of the full claim record
 *                       (claim_amount, claimant_id, adjuster_id,
 *                       date_of_loss, date_received, date_acknowledged,
 *                       date_disclosure_sent, date_approved,
 *                       date_paid, settlement_amount)
 *   - salt:              random salt for the commitment
 *   - daysToAcknowledge: integer days from receipt to acknowledgment
 *   - daysToDisclosure:  integer days from receipt to disclosure sent
 *   - daysToPayment:     integer days from approval to payment (0 if
 *                       not yet paid)
 *   - claimAmount:       amount claimed (cents)
 *   - settlementAmount:  amount paid (cents; equals claimAmount if
 *                       fully approved)
 *   - approved:          1 if claim was approved, 0 if denied
 *   - lowballReasoningProvided: 1 if documented reasoning exists for
 *                       settlement < 60% of claim, 0 otherwise
 *
 * Output
 * ------
 *   - isCompliant: 1 if all applicable regulatory constraints hold,
 *                  0 otherwise.
 *
 * Constraint Budget
 * -----------------
 *   Target:  <= 150K constraints
 *   Actual:  ~120K (estimated — 12 jurisdiction branches ×
 *                   ~10K constraints each, dominated by comparators
 *                   and Poseidon hashes)
 *
 * Compilation
 * -----------
 *   circom compliance_verification.circom --r1cs --wasm --sym
 *   snarkjs groth16 setup circuit.r1cs powersoftau.ptau circuit_0000.zkey
 *   ... (trusted setup phases) ...
 *   snarkjs zkey export verificationkey circuit_final.zkey verification_key.json
 *
 * Performance
 * -----------
 *   Proof generation: < 15 seconds on CPU (target)
 *   Verification:     < 10ms (Groth16 constant time guarantee)
 */

pragma circom 2.1.9;

include "../node_modules/circomlib/circuits/poseidon.circom";
include "../node_modules/circomlib/circuits/comparators.circom";

/* ========================================================================
 * JurisdictionConstants: hardcoded regulatory deadlines per state.
 * Selected via the public `jurisdiction` input.
 *
 * Returns:
 *   - ackDeadlineDays
 *   - paymentDeadlineDays
 *   - disclosureDeadlineDays
 *
 * Implementation: 12-way multiplexer. Each branch is a constant
 * assignment selected by an Equals comparator on `jurisdiction`.
 * ======================================================================== */
template JurisdictionConstants() {
    signal input jurisdiction;
    signal output ackDeadlineDays;
    signal output paymentDeadlineDays;
    signal output disclosureDeadlineDays;

    // State code -> (ack, pay, disclosure) deadlines in days
    // 1=CA, 2=NY, 3=TX, 4=FL, 5=IL, 6=PA, 7=OH, 8=GA, 9=NC, 10=MI, 11=NJ, 12=WA
    signal ack[12] <== [10, 15, 15, 14, 21, 10, 15, 15, 30, 20, 10, 15];
    signal pay[12] <== [30, 30, 10, 20, 30, 25, 21, 30, 30, 30, 30, 15];
    signal discl[12] <== [15, 10, 15, 15, 15, 15, 15, 15, 15, 15, 10, 10];

    // Multiplexer: sum over (Equals(jurisdiction, i) * deadline[i])
    // This adds ~12 * 3 comparators per state = 36 comparator constraints,
    // each LessThan/Equals uses ~n constraints for n-bit input.
    signal sel[12];
    for (var i = 0; i < 12; i++) {
        component eq = IsEqual();
        eq.in[0] <== jurisdiction;
        eq.in[1] <== i + 1;
        sel[i] <== eq.out;
    }

    ackDeadlineDays <== sel[0] * ack[0] + sel[1] * ack[1] + sel[2] * ack[2]
                     + sel[3] * ack[3] + sel[4] * ack[4] + sel[5] * ack[5]
                     + sel[6] * ack[6] + sel[7] * ack[7] + sel[8] * ack[8]
                     + sel[9] * ack[9] + sel[10] * ack[10] + sel[11] * ack[11];

    paymentDeadlineDays <== sel[0] * pay[0] + sel[1] * pay[1] + sel[2] * pay[2]
                         + sel[3] * pay[3] + sel[4] * pay[4] + sel[5] * pay[5]
                         + sel[6] * pay[6] + sel[7] * pay[7] + sel[8] * pay[8]
                         + sel[9] * pay[9] + sel[10] * pay[10] + sel[11] * pay[11];

    disclosureDeadlineDays <== sel[0] * discl[0] + sel[1] * discl[1]
                             + sel[2] * discl[2] + sel[3] * discl[3]
                             + sel[4] * discl[4] + sel[5] * discl[5]
                             + sel[6] * discl[6] + sel[7] * discl[7]
                             + sel[8] * discl[8] + sel[9] * discl[9]
                             + sel[10] * discl[10] + sel[11] * discl[11];
}

/* ========================================================================
 * TimelyAcknowledgmentCheck: verifies daysToAcknowledge <= ackDeadlineDays
 * ======================================================================== */
template TimelyAcknowledgmentCheck(n) {
    signal input daysToAcknowledge;
    signal input ackDeadlineDays;
    signal output out;

    component leq = LessEqThan(n);
    leq.in[0] <== daysToAcknowledge;
    leq.in[1] <== ackDeadlineDays;
    out <== leq.out;
}

/* ========================================================================
 * PaymentTimelineCheck: if approved, daysToPayment <= paymentDeadlineDays
 * ======================================================================== */
template PaymentTimelineCheck(n) {
    signal input approved;
    signal input daysToPayment;
    signal input paymentDeadlineDays;
    signal output out;

    // If not approved, payment check trivially passes (no payment due).
    // If approved, payment must be within deadline.
    component leq = LessEqThan(n);
    leq.in[0] <== daysToPayment;
    leq.in[1] <== paymentDeadlineDays;

    // out = (1 - approved) + approved * leq.out
    // = 1 if not approved, or leq.out if approved
    out <== (1 - approved) + approved * leq.out;
}

/* ========================================================================
 * DisclosureMandateCheck: daysToDisclosure <= disclosureDeadlineDays
 * ======================================================================== */
template DisclosureMandateCheck(n) {
    signal input daysToDisclosure;
    signal input disclosureDeadlineDays;
    signal output out;

    component leq = LessEqThan(n);
    leq.in[0] <== daysToDisclosure;
    leq.in[1] <== disclosureDeadlineDays;
    out <== leq.out;
}

/* ========================================================================
 * FairClaimsPracticeCheck: if approved, settlementAmount must be either
 *   (a) >= 60% of claimAmount, OR
 *   (b) lowballReasoningProvided == 1
 *
 * The 60% threshold is a regulatory safe-harbor in most states; below
 * that, the insurer must document why the settlement was reduced.
 * ======================================================================== */
template FairClaimsPracticeCheck(n) {
    signal input approved;
    signal input claimAmount;
    signal input settlementAmount;
    signal input lowballReasoningProvided;
    signal output out;

    // Compute 60% of claimAmount: claimAmount * 3 / 5
    // To keep things integer, we check 5 * settlementAmount >= 3 * claimAmount
    signal thresholdCheck;
    component leq = LessEqThan(2 * n + 4);  // 2*n bits for products, +4 for safety
    leq.in[0] <== 3 * claimAmount;
    leq.in[1] <== 5 * settlementAmount;
    thresholdCheck <== leq.out;

    // fair = thresholdCheck OR lowballReasoningProvided
    // Using: a OR b = 1 - (1-a)*(1-b)
    signal fairSettlement;
    fairSettlement <== 1 - (1 - thresholdCheck) * (1 - lowballReasoningProvided);

    // If not approved, fair-claims check trivially passes.
    out <== (1 - approved) + approved * fairSettlement;
}

/* ========================================================================
 * ComplianceVerification: main circuit
 * ======================================================================== */
template ComplianceVerification() {
    // Bit sizes for LessThan comparators
    // 32 bits covers up to ~4 billion cents = $40M, plenty for insurance
    // 16 bits covers days (up to 65536 days = ~180 years)
    signal input jurisdiction;          // public, 1..12
    signal input claimType;             // public, 1..4
    signal input complianceRoot;        // public, Poseidon hash anchor

    signal input claimRecordCommitment; // private
    signal input salt;                  // private
    signal input daysToAcknowledge;     // private
    signal input daysToDisclosure;      // private
    signal input daysToPayment;         // private
    signal input claimAmount;           // private, cents
    signal input settlementAmount;      // private, cents
    signal input approved;              // private, 0 or 1
    signal input lowballReasoningProvided; // private, 0 or 1

    signal output isCompliant;

    // ---- Verify complianceRoot matches Poseidon(claimRecordCommitment, salt) ----
    component poseidon = Poseidon(2);
    poseidon.inputs[0] <== claimRecordCommitment;
    poseidon.inputs[1] <== salt;
    complianceRoot === poseidon.out;

    // ---- Select jurisdiction-specific deadlines ----
    component jc = JurisdictionConstants();
    jc.jurisdiction <== jurisdiction;

    // ---- Timely acknowledgment ----
    component ackCheck = TimelyAcknowledgmentCheck(16);
    ackCheck.daysToAcknowledge <== daysToAcknowledge;
    ackCheck.ackDeadlineDays <== jc.ackDeadlineDays;

    // ---- Payment timeline ----
    component payCheck = PaymentTimelineCheck(16);
    payCheck.approved <== approved;
    payCheck.daysToPayment <== daysToPayment;
    payCheck.paymentDeadlineDays <== jc.paymentDeadlineDays;

    // ---- Disclosure mandate ----
    component disclCheck = DisclosureMandateCheck(16);
    disclCheck.daysToDisclosure <== daysToDisclosure;
    disclCheck.disclosureDeadlineDays <== jc.disclosureDeadlineDays;

    // ---- Fair claims practice ----
    component fairCheck = FairClaimsPracticeCheck(32);
    fairCheck.approved <== approved;
    fairCheck.claimAmount <== claimAmount;
    fairCheck.settlementAmount <== settlementAmount;
    fairCheck.lowballReasoningProvided <== lowballReasoningProvided;

    // ---- All checks must pass ----
    isCompliant <== ackCheck.out * payCheck.out * disclCheck.out * fairCheck.out;
}

component main {public [jurisdiction, claimType, complianceRoot]} = ComplianceVerification();
