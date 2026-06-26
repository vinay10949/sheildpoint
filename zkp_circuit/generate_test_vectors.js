/**
 * ShieldPoint ZKP Test Vector Generator
 * ======================================
 * Generates and verifies test vectors for the Policy Validity Proof circuit
 * covering 5+ scenarios as required by the acceptance criteria:
 *
 *   1. Valid policy (all conditions pass)
 *   2. Expired policy (date of loss after expiration)
 *   3. Uncovered peril (perilType not in perils list)
 *   4. Over-limit claim (claimAmount > coverageLimit)
 *   5. Wrong commitment (policyCommitment doesn't match)
 *   6. Inactive policy (policyStatus = 0)
 *   7. Date before effective date
 */

const snarkjs = require("snarkjs");
const fs = require("fs");
const path = require("path");
const circomlibjs = require("circomlibjs");

// Peril type code mapping
const PERIL_CODES = {
    wind: 1,
    hail: 2,
    fire: 3,
    theft: 4,
    vandalism: 5,
    lightning: 6,
    collision: 7,
    comprehensive: 8,
    flood: 9,       // typically excluded
    earthquake: 10, // typically excluded
};

// Date to days since Unix epoch
function dateToDays(dateStr) {
    const epoch = new Date("1970-01-01");
    const target = new Date(dateStr);
    return Math.floor((target - epoch) / (1000 * 60 * 60 * 24));
}

// Build Poseidon hasher lazily
let poseidonHasher = null;

async function getPoseidon() {
    if (!poseidonHasher) {
        poseidonHasher = await circomlibjs.buildPoseidon();
    }
    return poseidonHasher;
}

// Compute Poseidon hash for policy commitment
async function computePoseidonHash(inputs) {
    const poseidon = await getPoseidon();
    const hash = poseidon(inputs);
    return poseidon.F.toString(hash);
}

// Convert to string for JSON serialization
function bigIntToStr(val) {
    return val.toString();
}

// Build circuit input from policy/claim data
async function buildCircuitInput(scenario) {
    const { policy, claim, expectedValid } = scenario;

    // Compute policy commitment
    const policyCommitment = await computePoseidonHash([
        BigInt(policy.policyId),
        BigInt(policy.salt),
    ]);

    // Map perils to codes and pad to 8 slots
    const perilCodes = policy.perilsCovered.map(p => PERIL_CODES[p] || 0);
    while (perilCodes.length < 8) {
        perilCodes.push(0);
    }

    const input = {
        // Public inputs
        policyCommitment: policyCommitment,  // Already a string from Poseidon
        claimType: String(PERIL_CODES[claim.perilType] || 0),

        // Private inputs: Policy
        policyId: String(policy.policyId),
        salt: String(policy.salt),
        coverageLimit: String(policy.coverageLimit),
        deductible: String(policy.deductible),
        effectiveDate: String(dateToDays(policy.effectiveDate)),
        expirationDate: String(dateToDays(policy.expirationDate)),
        perils: perilCodes.map(String),
        exclusions: (policy.perilsExcluded || ["flood", "earthquake", "wear_and_tear", "mold", "intentional_damage"])
            .map(p => String(PERIL_CODES[p] || 0)).concat(Array(8).fill("0")).slice(0, 8),
        policyStatus: String(policy.status === "active" ? 1 : 0),

        // Private inputs: Claim
        claimAmount: String(claim.amount),
        perilType: String(PERIL_CODES[claim.perilType] || 0),
        dateOfLoss: String(dateToDays(claim.dateOfLoss)),
    };

    return { input, expectedValid, scenario: scenario.name };
}

async function main() {
    const buildDir = path.join(__dirname, "build", "policy_validity_js");
    const wasmPath = path.join(buildDir, "policy_validity.wasm");
    const zkeyPath = path.join(__dirname, "keys", "circuit_final.zkey");
    const vkeyPath = path.join(__dirname, "keys", "verification_key.json");
    const tvDir = path.join(__dirname, "test_vectors");

    // Ensure test_vectors directory exists
    if (!fs.existsSync(tvDir)) {
        fs.mkdirSync(tvDir, { recursive: true });
    }

    const vkey = JSON.parse(fs.readFileSync(vkeyPath, "utf8"));

    // Define test scenarios
    const scenarios = [
        {
            name: "valid_policy",
            description: "Valid active policy with covered peril, date in range, amount within limit",
            policy: {
                policyId: 1001,
                salt: 42,
                coverageLimit: 250000,
                deductible: 1000,
                effectiveDate: "2024-01-01",
                expirationDate: "2027-01-01",
                perilsCovered: ["wind", "hail", "fire", "theft", "vandalism", "lightning"],
                status: "active",
            },
            claim: {
                amount: 1250,
                perilType: "wind",
                dateOfLoss: "2026-03-14",
            },
            expectedValid: true,
        },
        {
            name: "expired_policy",
            description: "Policy expired - date of loss is after expiration date",
            policy: {
                policyId: 2002,
                salt: 99,
                coverageLimit: 100000,
                deductible: 1000,
                effectiveDate: "2022-01-01",
                expirationDate: "2023-01-01",
                perilsCovered: ["wind", "hail", "fire"],
                status: "active",
            },
            claim: {
                amount: 5000,
                perilType: "wind",
                dateOfLoss: "2026-03-14",  // After expiration
            },
            expectedValid: false,
        },
        {
            name: "uncovered_peril",
            description: "Peril type (flood) is not in the covered perils list",
            policy: {
                policyId: 3003,
                salt: 77,
                coverageLimit: 250000,
                deductible: 1000,
                effectiveDate: "2024-01-01",
                expirationDate: "2027-01-01",
                perilsCovered: ["wind", "hail", "fire", "theft", "vandalism", "lightning"],
                status: "active",
            },
            claim: {
                amount: 5000,
                perilType: "flood",  // NOT covered
                dateOfLoss: "2026-03-14",
            },
            expectedValid: false,
        },
        {
            name: "over_limit_claim",
            description: "Claim amount exceeds the coverage limit",
            policy: {
                policyId: 4004,
                salt: 55,
                coverageLimit: 50000,
                deductible: 500,
                effectiveDate: "2024-01-01",
                expirationDate: "2027-01-01",
                perilsCovered: ["collision", "comprehensive", "wind"],
                status: "active",
            },
            claim: {
                amount: 75000,  // Exceeds 50000 limit
                perilType: "collision",
                dateOfLoss: "2026-04-02",
            },
            expectedValid: false,
        },
        {
            name: "wrong_commitment",
            description: "Policy commitment does not match - wrong policyId/salt",
            policy: {
                policyId: 5005,  // Different from what commitment was computed for
                salt: 33,
                coverageLimit: 250000,
                deductible: 1000,
                effectiveDate: "2024-01-01",
                expirationDate: "2027-01-01",
                perilsCovered: ["wind", "hail", "fire", "theft"],
                status: "active",
            },
            claim: {
                amount: 5000,
                perilType: "wind",
                dateOfLoss: "2026-03-14",
            },
            expectedValid: false,
            // Override: use wrong commitment
            overrideCommitment: true,
        },
        {
            name: "inactive_policy",
            description: "Policy status is inactive (0) instead of active (1)",
            policy: {
                policyId: 6006,
                salt: 88,
                coverageLimit: 250000,
                deductible: 1000,
                effectiveDate: "2024-01-01",
                expirationDate: "2027-01-01",
                perilsCovered: ["wind", "hail", "fire", "theft"],
                status: "inactive",
            },
            claim: {
                amount: 5000,
                perilType: "wind",
                dateOfLoss: "2026-03-14",
            },
            expectedValid: false,
        },
        {
            name: "date_before_effective",
            description: "Date of loss is before the policy effective date",
            policy: {
                policyId: 7007,
                salt: 11,
                coverageLimit: 250000,
                deductible: 1000,
                effectiveDate: "2024-01-01",
                expirationDate: "2027-01-01",
                perilsCovered: ["wind", "hail", "fire", "theft"],
                status: "active",
            },
            claim: {
                amount: 5000,
                perilType: "wind",
                dateOfLoss: "2023-06-01",  // Before effective date
            },
            expectedValid: false,
        },
    ];

    const results = [];

    for (const scenario of scenarios) {
        console.log(`\n=== Generating test vector: ${scenario.name} ===`);
        console.log(`  ${scenario.description}`);

        const { input, expectedValid } = await buildCircuitInput(scenario);

        // Handle wrong commitment scenario: compute commitment with different inputs
        if (scenario.overrideCommitment) {
            const wrongCommitment = await computePoseidonHash([BigInt(9999), BigInt(0)]);
            input.policyCommitment = wrongCommitment;
        }

        // Save input
        const inputPath = path.join(tvDir, `${scenario.name}_input.json`);
        fs.writeFileSync(inputPath, JSON.stringify(input, null, 2));

        // Generate witness
        console.log("  Calculating witness...");
        const witnessPath = path.join(tvDir, `${scenario.name}_witness.wtns`);
        await snarkjs.wtns.calculate(
            input,
            wasmPath,
            witnessPath
        );

        // Generate proof
        console.log("  Generating Groth16 proof...");
        const startTime = Date.now();
        const { proof, publicSignals } = await snarkjs.groth16.prove(
            zkeyPath,
            witnessPath
        );
        const proofTimeMs = Date.now() - startTime;

        // Save proof and public signals
        fs.writeFileSync(
            path.join(tvDir, `${scenario.name}_proof.json`),
            JSON.stringify(proof, null, 2)
        );
        fs.writeFileSync(
            path.join(tvDir, `${scenario.name}_public.json`),
            JSON.stringify(publicSignals, null, 2)
        );

        // Verify proof
        const verifyStart = Date.now();
        const verified = await snarkjs.groth16.verify(vkey, publicSignals, proof);
        const verifyTimeMs = Date.now() - verifyStart;

        // Check the isValid output (last public signal)
        const isValidSignal = publicSignals[0]; // First public signal is isValid
        const actualValid = isValidSignal === "1";

        console.log(`  Expected valid: ${expectedValid}`);
        console.log(`  Actual valid:   ${actualValid}`);
        console.log(`  Proof verified: ${verified}`);
        console.log(`  Proof time:     ${proofTimeMs}ms`);
        console.log(`  Verify time:    ${verifyTimeMs}ms`);

        const pass = actualValid === expectedValid && verified;
        console.log(`  RESULT: ${pass ? "PASS" : "FAIL"}`);

        results.push({
            scenario: scenario.name,
            description: scenario.description,
            expectedValid,
            actualValid,
            proofVerified: verified,
            proofTimeMs,
            verifyTimeMs,
            pass,
        });
    }

    // Save summary
    const summaryPath = path.join(tvDir, "summary.json");
    fs.writeFileSync(summaryPath, JSON.stringify(results, null, 2));

    console.log("\n\n=== TEST VECTOR SUMMARY ===");
    for (const r of results) {
        console.log(`  ${r.pass ? "PASS" : "FAIL"}: ${r.scenario} (valid=${r.actualValid}, proof_ok=${r.proofVerified}, prove=${r.proofTimeMs}ms, verify=${r.verifyTimeMs}ms)`);
    }

    const allPass = results.every(r => r.pass);
    console.log(`\n  Overall: ${allPass ? "ALL TESTS PASSED" : "SOME TESTS FAILED"}`);
    process.exit(allPass ? 0 : 1);
}

main().catch(e => {
    console.error("Error generating test vectors:", e);
    process.exit(1);
});
