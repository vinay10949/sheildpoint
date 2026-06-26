# ShieldPoint ZKP Policy Validity Proof Circuit

Zero-knowledge proof system for insurance policy validity verification using Circom 2.1.9 and SnarkJS (Groth16 over BN-128 curve).

## Overview

This circuit enables an insurance claimant to prove that their claim is valid under a given policy **without revealing** the underlying policy details, claim amount, or peril information. The verifier only learns:

- Whether the claim is valid (`isValid = 1` or `0`)
- The policy commitment (Poseidon hash of `policyId` + `salt`)
- The claim type (peril code)

### What the Circuit Proves

The circuit enforces **7 constraints** combined via AND reduction:

| # | Constraint | Description |
|---|-----------|-------------|
| 1 | Commitment Verification | `Poseidon(policyId, salt) == policyCommitment` |
| 2 | Policy Status | `policyStatus == 1` (active) |
| 3 | Peril Membership | `perilType ∈ perils[8]` (covered peril) |
| 4 | Peril Exclusion | `perilType ∉ exclusions[8]` (not excluded) |
| 5 | Date Range | `effectiveDate ≤ dateOfLoss ≤ expirationDate` |
| 6 | Coverage Limit | `claimAmount ≤ coverageLimit` |
| 7 | Deductible | `claimAmount ≥ deductible` |

### Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| R1CS Constraints | < 50,000 | **556** |
| Proof Generation | < 5,000 ms | **90–340 ms** |
| Verification | < 10 ms | **10–21 ms** |
| Trusted Setup Contributors | ≥ 3 | **8** |

## Architecture

```
zkp_circuit/
├── circuits/
│   └── policy_validity.circom    # Main circuit (v2) with 7 constraints
├── build/
│   ├── policy_validity.r1cs      # Compiled R1CS constraint system
│   ├── policy_validity.sym       # Symbol file for debugging
│   ├── policy_validity_js/
│   │   ├── policy_validity.wasm  # WASM witness calculator
│   │   ├── witness_calculator.js
│   │   └── generate_witness.js
│   └── verifier.sol              # Solidity Groth16 verifier (182 lines)
├── keys/
│   ├── pot12_0000.ptau ... pot12_0008.ptau  # Powers of Tau (Phase 1)
│   ├── pot12_final.ptau                       # Final Phase 1 transcript
│   ├── circuit_0000.zkey ... circuit_0008.zkey  # Circuit zkeys (Phase 2)
│   ├── circuit_final.zkey                       # Final proving key
│   └── verification_key.json                    # Verification key
├── test_vectors/                 # Generated test vectors (7 scenarios)
│   ├── *_input.json, *_public.json, *_proof.json, *_witness.wtns
│   └── summary.json
├── zkp_prover.py                 # Python ZKPProver class (SnarkJS wrapper)
├── test_integration.py           # 20 integration tests (9 scenarios)
├── generate_test_vectors.js      # Node.js test vector generator
├── Makefile                      # Build automation
├── package.json                  # npm dependencies (circomlib, circomlibjs)
└── README.md                     # This file

tool_registry/
├── shieldpoint/
│   ├── zkp.py                    # Dual-mode integration (Groth16 + stub fallback)
│   ├── tools.py                  # Tool registry with zkp_prove_policy / zkp_verify_proof
│   ├── tool_registry.py          # Core ToolRegistry class
│   ├── state_machine.py          # Claim state machine
│   ├── db.py                     # Repository layer
│   ├── langfuse_span.py          # Observability spans
│   └── __init__.py
├── tests/
│   ├── test_zkp_tools.py         # 19 dual-mode tests
│   ├── test_tool_registry.py
│   ├── conftest.py
│   └── ...
├── pyproject.toml
└── requirements.txt
```

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| **Circom** | 2.1.9 | `curl -L -o /usr/local/bin/circom https://github.com/iden3/circom/releases/download/v2.1.9/circom-linux-amd64 && chmod +x /usr/local/bin/circom` |
| **SnarkJS** | latest | `npm install -g snarkjs` |
| **Node.js** | ≥ 18 | `nvm install 18 && nvm use 18` |
| **Python** | ≥ 3.11 | System install |
| **pytest** | ≥ 7.0 | `pip install pytest` |

## Quick Start

### 1. Install Dependencies

```bash
cd zkp_circuit/
make setup
```

This installs:
- Circom 2.1.9 compiler
- SnarkJS (global)
- circomlib + circomlibjs (npm local)

### 2. Compile the Circuit

```bash
make compile
```

This produces:
- `build/policy_validity.r1cs` — R1CS constraint system
- `build/policy_validity_js/policy_validity.wasm` — WASM witness calculator
- `build/policy_validity.sym` — Debug symbols

### 3. Run Trusted Setup

```bash
make trusted-setup
```

This performs:
1. **Phase 1**: Powers of Tau ceremony with 8 contributors (bn128, 2^12)
2. **Phase 2**: Circuit-specific setup with 8 contributions
3. Outputs `keys/circuit_final.zkey` (proving key) and `keys/verification_key.json`

> **Note**: The trusted setup takes ~2–5 minutes. For production, replace with a multi-party ceremony.

### 4. Run Tests

```bash
# Integration tests (20 tests, 9 scenarios)
make test

# All tests (integration + tool_registry)
make test-all

# Generate and verify test vectors (7 scenarios)
make test-vectors
```

### 5. Export Solidity Verifier

```bash
make solidity
```

Outputs `build/verifier.sol` — a deployable Groth16 verifier contract for Ethereum/EVM chains.

## Usage

### Python: ZKPProver

```python
from zkp_prover import ZKPProver

prover = ZKPProver()

# Generate a proof
result = prover.prove(
    policy_id=1001,
    salt=42,
    coverage_limit=250000,
    deductible=1000,
    effective_date="2024-01-01",
    expiration_date="2027-01-01",
    perils=[1, 2, 3, 4, 5, 6, 0, 0],    # wind, hail, fire, theft, vandalism, lightning
    exclusions=[9, 10, 12, 13, 14, 0, 0, 0],  # flood, earthquake, wear, mold, intentional
    policy_status=1,
    claim_amount=1250,
    peril_type=1,   # wind
    date_of_loss="2026-03-14",
)

print(f"Valid: {result['verified']}")         # True
print(f"Latency: {result['prover_latency_ms']:.1f}ms")

# Verify the proof
verify_result = prover.verify(result["proof"], result["public_signals"])
print(f"Verified: {verify_result['verified']}")  # True
```

### Python: Tool Registry (Dual-Mode)

```python
from shieldpoint import zkp_prove_policy, zkp_verify_proof

# Generate proof (auto-detects Groth16 or falls back to stub)
result = zkp_prove_policy(
    claim_id="CLM-2026-0001",
    policy_id="HO-2024-001",
    claim_amount=1250.00,
    coverage_limit=250_000,
    peril_covered=True,
    policy_active=True,
)

# Verify proof
verify = zkp_verify_proof(
    proof=result["proof"],
    public_signals=result["public_signals"],
)
```

The tool registry module (`shieldpoint/zkp.py`) automatically detects whether compiled circuit artifacts are available. If found, it uses real Groth16 proofs; otherwise, it falls back to a SHA-256 stub for development/testing.

### JavaScript: SnarkJS Direct

```javascript
const snarkjs = require("snarkjs");

// Generate proof
const { proof, publicSignals } = await snarkjs.groth16.fullprove(
    input,
    "build/policy_validity_js/policy_validity.wasm",
    "keys/circuit_final.zkey"
);

// Verify proof
const vkey = JSON.parse(fs.readFileSync("keys/verification_key.json"));
const verified = await snarkjs.groth16.verify(vkey, publicSignals, proof);
```

### Solidity: On-Chain Verification

```solidity
// Deploy the verifier
Groth16Verifier verifier = new Groth16Verifier();

// Verify a proof on-chain
bool isValid = verifier.verifyProof(
    [_pA[0], _pA[1]],           // uint[2]
    [[_pB[0][0], _pB[0][1]],    // uint[2][2]
     [_pB[1][0], _pB[1][1]]],
    [_pC[0], _pC[1]],           // uint[2]
    [isValid, commitment, claimType]  // uint[3] public signals
);
```

## Test Scenarios

The test suite covers **9 scenarios** across **39 tests**:

| Scenario | Description | Expected `isValid` |
|----------|-------------|--------------------:|
| Valid Policy | All conditions met | 1 |
| Expired Policy | Date of loss after expiration | 0 |
| Uncovered Peril | Peril type not in covered list | 0 |
| Over-Limit Claim | `claimAmount > coverageLimit` | 0 |
| Wrong Commitment | `policyCommitment ≠ Poseidon(policyId, salt)` | 0 |
| Inactive Policy | `policyStatus = 0` | 0 |
| Date Before Effective | Loss date before policy start | 0 |
| Excluded Peril (v2) | Peril in exclusion list | 0 |
| Below Deductible (v2) | `claimAmount < deductible` | 0 |

## Peril Codes

| Code | Peril | Code | Peril |
|------|-------|------|-------|
| 1 | Wind | 8 | Comprehensive |
| 2 | Hail | 9 | Flood (typically excluded) |
| 3 | Fire | 10 | Earthquake (typically excluded) |
| 4 | Theft | 11 | Uninsured Motorist |
| 5 | Vandalism | 12 | Wear and Tear (excluded) |
| 6 | Lightning | 13 | Mold (excluded) |
| 7 | Collision | 14 | Intentional Damage (excluded) |

## Circuit Design

### Public Inputs
- `policyCommitment` — Poseidon hash of `(policyId, salt)`
- `claimType` — Numeric peril identifier (must equal `perilType`)

### Private Inputs
- `policyId`, `salt` — For commitment verification
- `coverageLimit`, `deductible` — Policy financial parameters
- `effectiveDate`, `expirationDate` — Policy date range (days since epoch)
- `perils[8]` — Covered peril type codes (0 = unused slot)
- `exclusions[8]` — Excluded peril type codes (0 = unused slot)
- `policyStatus` — 1 = active, 0 = inactive
- `claimAmount` — Amount being claimed
- `perilType` — Peril code of the claim event
- `dateOfLoss` — Date of the loss event (days since epoch)

### Templates

| Template | Purpose | Technique |
|----------|---------|-----------|
| `RangeCheck(n)` | Date range verification | `LessEqThan` × 2 + AND |
| `PerilMembership(n)` | Peril in covered list | `IsEqual` × n + OR reduction |
| `PerilExclusion(n)` | Peril NOT in exclusion list | `IsEqual` × n + OR + NOT |
| `DeductibleCheck(n)` | Claim ≥ deductible | `GreaterEqThan` |
| `PolicyValidityProof()` | Main circuit | AND of all 7 checks |

## Security Considerations

- **Trusted Setup**: The current setup uses simulated contributions. For production, conduct a real multi-party computation (MPC) ceremony with independent participants.
- **Poseidon Hash**: SNARK-friendly hash prevents commitment collisions within the BN-128 field.
- **Soundness**: All constraints are enforced as R1CS quadratic equations. The AND reduction (`a * b`) ensures all conditions must hold simultaneously.
- **Zero Knowledge**: Private inputs (policyId, salt, financials, perils) are never exposed in the proof or public signals.
- **On-Chain**: The Solidity verifier enables trustless verification on Ethereum/EVM chains without relying on the prover.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `circom: command not found` | Run `make setup` or install Circom manually |
| `snarkjs: command not found` | Run `npm install -g snarkjs` |
| `FileNotFoundError: Circuit WASM not found` | Run `make compile` then `make trusted-setup` |
| `Poseidon computation failed` | Run `npm install circomlibjs` in `zkp_circuit/` |
| Proof generation timeout | Increase subprocess timeout (default: 60s) |
| Verification fails with valid proof | Ensure `_circuit_public` array matches actual SnarkJS output |

## Make Targets

```
make setup          Install Circom + SnarkJS + npm deps
make compile        Compile circuit to R1CS + WASM
make trusted-setup  Full Groth16 trusted setup (8 contributors)
make test-vectors   Generate and verify test vectors
make test           Run Python integration tests
make test-all       Run all tests (integration + tool_registry)
make solidity       Export Solidity verifier contract
make clean          Remove build artifacts (keeps keys)
make clean-all      Remove everything including keys
make info           Show build configuration
```

## License

This project is part of the ShieldPoint Claims Automation Platform.
