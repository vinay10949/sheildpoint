"""
ShieldPoint ZKP Compliance Verification Prover
==============================================

Python wrapper for the Circom ``compliance_verification.circom`` circuit.
Generates and verifies zk-SNARK proofs that an insurance claim was
processed in compliance with the regulations of one of ShieldPoint's 12
operating states — WITHOUT revealing any sensitive claim data.

This is the second ZKP gate in the state machine
(``ZKP_COMPLIANCE_PROOF`` state), and is more complex than the Policy
Validity Proof (~120K constraints vs. ~50K) because it encodes the
specific regulations of each of the 12 states.

Architecture
------------
- :class:`ComplianceProver` — high-level prover/verifier. If the
  compiled Circom circuit artifacts are available (WASM + zkey + vkey),
  it shells out to SnarkJS to generate a real Groth16 proof. Otherwise,
  it falls back to a deterministic Python stub that re-evaluates the
  same constraints as the circuit and returns a structured result.
- :class:`TraditionalComplianceChecker` — plain-Python implementation
  of the same regulatory checks, run in parallel with the ZKP path for
  the first 12 months after deployment. Writes a
  ``traditional_compliance_result`` row alongside the ZKP result.
- :class:`ComplianceClaimRecord` — dataclass representing the private
  inputs to the circuit (claim record).

The 12 ShieldPoint Operating States
-----------------------------------
1=CA, 2=NY, 3=TX, 4=FL, 5=IL, 6=PA, 7=OH, 8=GA, 9=NC, 10=MI, 11=NJ, 12=WA

Each state has three regulatory deadlines encoded in the circuit:
- Acknowledgment deadline (days from receipt to formal acknowledgment)
- Payment deadline (days from approval to payment)
- Disclosure deadline (days from receipt to sending required disclosures)

Plus the fair-claims-practice constraint (settlement must be ≥ 60% of
claim amount, OR documented reasoning must be provided).

Performance Targets
-------------------
- Proof generation: < 15 seconds on CPU
- Verification: < 10ms (Groth16 constant time guarantee)
- Constraint count: ≤ 150K (target), ~120K (actual)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("shieldpoint.compliance_prover")


# ===========================================================================
# 12 operating states and their regulatory deadlines
# ===========================================================================
@dataclass(frozen=True)
class StateRegulation:
    """Regulatory deadlines for one operating state."""
    code: int          # 1..12 (circuit input)
    abbr: str          # "CA", "NY", ...
    name: str
    ack_deadline_days: int
    payment_deadline_days: int
    disclosure_deadline_days: int


STATE_REGULATIONS: dict[str, StateRegulation] = {
    "CA": StateRegulation(1, "CA", "California", 10, 30, 15),
    "NY": StateRegulation(2, "NY", "New York", 15, 30, 10),
    "TX": StateRegulation(3, "TX", "Texas", 15, 10, 15),
    "FL": StateRegulation(4, "FL", "Florida", 14, 20, 15),
    "IL": StateRegulation(5, "IL", "Illinois", 21, 30, 15),
    "PA": StateRegulation(6, "PA", "Pennsylvania", 10, 25, 15),
    "OH": StateRegulation(7, "OH", "Ohio", 15, 21, 15),
    "GA": StateRegulation(8, "GA", "Georgia", 15, 30, 15),
    "NC": StateRegulation(9, "NC", "North Carolina", 30, 30, 15),
    "MI": StateRegulation(10, "MI", "Michigan", 20, 30, 15),
    "NJ": StateRegulation(11, "NJ", "New Jersey", 10, 30, 10),
    "WA": StateRegulation(12, "WA", "Washington", 15, 15, 10),
}

# Reverse lookup: code -> abbr
CODE_TO_ABBR: dict[int, str] = {r.code: r.abbr for r in STATE_REGULATIONS.values()}

# Claim type codes (matches the circuit's public claimType input)
CLAIM_TYPE_CODES: dict[str, int] = {
    "property_damage": 1,
    "auto": 2,
    "liability": 3,
    "medical": 4,
}


# ===========================================================================
# Compliance claim record (private inputs)
# ===========================================================================
@dataclass
class ComplianceClaimRecord:
    """Private inputs to the compliance verification circuit.

    All date fields are integers: days since epoch (1970-01-01) for date
    math, OR days-since-receipt for relative timing. We use
    days-since-receipt in this Python wrapper because it's simpler and
    matches the circuit's input shape.
    """
    claim_record_commitment: int   # Poseidon hash of the full claim record
    salt: int                       # random salt for the commitment
    jurisdiction_code: int          # 1..12 (matches circuit public input)
    claim_type_code: int            # 1..4
    days_to_acknowledge: int
    days_to_disclosure: int
    days_to_payment: int            # 0 if not yet paid
    claim_amount_cents: int
    settlement_amount_cents: int
    approved: int                   # 0 or 1
    lowball_reasoning_provided: int # 0 or 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ===========================================================================
# Result types
# ===========================================================================
@dataclass
class ComplianceProofResult:
    """Result of a compliance proof generation."""
    verified: bool
    statement: str
    proof_type: str    # "groth16" or "stub"
    jurisdiction: str
    claim_type: str
    compliance_root: Optional[str] = None
    proof: Optional[str] = None
    public_signals: Optional[dict[str, Any]] = None
    prover_latency_ms: float = 0.0
    checks: dict[str, bool] = field(default_factory=dict)
    """Per-check pass/fail: ack, payment, disclosure, fair_claims_practice."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComplianceVerifyResult:
    """Result of a compliance proof verification."""
    verified: bool
    verifier: str     # "groth16" or "stub"
    verification_key: str
    latency_ms: float
    reason: str


# ===========================================================================
# Traditional (non-ZKP) compliance checker — parallel path for first 12 months
# ===========================================================================
class TraditionalComplianceChecker:
    """Plain-Python implementation of the same regulatory checks the
    circuit encodes. Runs in parallel with the ZKP path for the first
    12 months after deployment. If the two disagree, an alert is raised.

    This class is intentionally simple and dependency-free so it can be
    audited by regulators independently of the ZKP circuit.
    """

    FAIR_CLAIMS_THRESHOLD_PCT = 0.60  # settlement must be >= 60% of claim

    def check(self, record: ComplianceClaimRecord) -> dict[str, Any]:
        """Run all four regulatory checks. Returns a dict with per-check
        results and an overall ``compliant`` boolean."""
        reg = STATE_REGULATIONS.get(CODE_TO_ABBR.get(record.jurisdiction_code, ""))
        if reg is None:
            return {
                "compliant": False,
                "reason": f"Unknown jurisdiction code: {record.jurisdiction_code}",
                "checks": {},
            }
        checks = {
            "timely_acknowledgment":
                record.days_to_acknowledge <= reg.ack_deadline_days,
            "payment_timeline": (
                # If not approved, payment check trivially passes
                record.approved == 0
                or record.days_to_payment <= reg.payment_deadline_days
            ),
            "disclosure_mandate":
                record.days_to_disclosure <= reg.disclosure_deadline_days,
            "fair_claims_practice": self._fair_claims_check(record),
        }
        compliant = all(checks.values())
        return {
            "compliant": compliant,
            "reason": (
                "All regulatory checks passed."
                if compliant else
                f"Failed checks: {[k for k,v in checks.items() if not v]}"
            ),
            "checks": checks,
            "jurisdiction": reg.abbr,
            "jurisdiction_name": reg.name,
            "deadlines": {
                "ack_deadline_days": reg.ack_deadline_days,
                "payment_deadline_days": reg.payment_deadline_days,
                "disclosure_deadline_days": reg.disclosure_deadline_days,
            },
        }

    def _fair_claims_check(self, record: ComplianceClaimRecord) -> bool:
        """If approved, settlement must be ≥ 60% of claim OR
        lowball_reasoning_provided must be 1."""
        if record.approved == 0:
            return True
        if record.claim_amount_cents <= 0:
            return True
        ratio = record.settlement_amount_cents / record.claim_amount_cents
        return ratio >= self.FAIR_CLAIMS_THRESHOLD_PCT or \
               record.lowball_reasoning_provided == 1


# ===========================================================================
# Main prover class
# ===========================================================================
class ComplianceProver:
    """Generates and verifies ZKP compliance proofs.

    If the compiled Circom circuit artifacts (WASM, zkey, vkey) are
    available at ``circuit_dir``, this class shells out to SnarkJS to
    generate a real Groth16 proof. Otherwise, it falls back to a
    deterministic Python stub that re-evaluates the same constraints as
    the circuit and returns a structured result.

    The fallback path is cryptographically meaningless but preserves
    the API contract so the rest of the agent stack (state machine,
    Langfuse spans, integration tests) continues to work end-to-end
    without a compiled circuit.
    """

    CIRCUIT_NAME = "compliance_verification"
    FAIR_CLAIMS_THRESHOLD_PCT = 0.60

    def __init__(self, *, circuit_dir: Optional[Path] = None) -> None:
        self.circuit_dir = circuit_dir or self._find_circuit_dir()
        self._traditional = TraditionalComplianceChecker()
        self._has_real_circuit = self._check_artifacts()

    @staticmethod
    def _find_circuit_dir() -> Optional[Path]:
        """Locate the zkp_circuit/ directory containing the compiled
        compliance circuit artifacts."""
        candidates = [
            Path(__file__).resolve().parent.parent,
            Path(__file__).resolve().parent.parent.parent,
        ]
        for c in candidates:
            wasm = c / "build" / "compliance_verification_js" / \
                   "compliance_verification.wasm"
            zkey = c / "keys" / "compliance_circuit_final.zkey"
            vkey = c / "keys" / "compliance_verification_key.json"
            if wasm.exists() and zkey.exists() and vkey.exists():
                return c
        return None

    def _check_artifacts(self) -> bool:
        if self.circuit_dir is None:
            return False
        wasm = self.circuit_dir / "build" / "compliance_verification_js" / \
               "compliance_verification.wasm"
        zkey = self.circuit_dir / "keys" / "compliance_circuit_final.zkey"
        vkey = self.circuit_dir / "keys" / "compliance_verification_key.json"
        return wasm.exists() and zkey.exists() and vkey.exists()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def prove(self, record: ComplianceClaimRecord) -> ComplianceProofResult:
        """Generate a compliance proof for the given claim record.

        If the real Groth16 circuit is available, generates a real proof.
        Otherwise, falls back to the stub prover that re-evaluates the
        constraints in Python.
        """
        if self._has_real_circuit:
            return self._prove_real(record)
        return self._prove_stub(record)

    def verify(self, proof: dict[str, Any] | str,
               public_signals: dict[str, Any] | list[Any]) -> ComplianceVerifyResult:
        """Verify a compliance proof.

        Returns a :class:`ComplianceVerifyResult` with ``verified=True``
        if the proof is valid.
        """
        if self._has_real_circuit:
            return self._verify_real(proof, public_signals)
        return self._verify_stub(proof, public_signals)

    def traditional_check(self, record: ComplianceClaimRecord) -> dict[str, Any]:
        """Run the traditional (non-ZKP) compliance check in parallel.

        Per the acceptance criteria, this path runs alongside the ZKP
        path for the first 12 months after deployment. Divergences
        between the two paths raise alerts.
        """
        return self._traditional.check(record)

    # ------------------------------------------------------------------ #
    # Real Groth16 prover/verifier (shells out to snarkjs)
    # ------------------------------------------------------------------ #
    def _prove_real(self, record: ComplianceClaimRecord) -> ComplianceProofResult:
        """Generate a real Groth16 proof via snarkjs."""
        started = time.perf_counter()
        wasm = self.circuit_dir / "build" / "compliance_verification_js" / \
               "compliance_verification.wasm"
        zkey = self.circuit_dir / "keys" / "compliance_circuit_final.zkey"
        vkey = self.circuit_dir / "keys" / "compliance_verification_key.json"

        # Compute complianceRoot = Poseidon(claim_record_commitment, salt)
        # via snarkjs or a JS helper. For simplicity here, we use a
        # SHA-256 placeholder — production wires up Poseidon via circomlibjs.
        compliance_root = self._compute_compliance_root_stub(record)

        # Build the circuit input JSON
        circuit_input = {
            "jurisdiction": str(record.jurisdiction_code),
            "claimType": str(record.claim_type_code),
            "complianceRoot": str(compliance_root),
            "claimRecordCommitment": str(record.claim_record_commitment),
            "salt": str(record.salt),
            "daysToAcknowledge": str(record.days_to_acknowledge),
            "daysToDisclosure": str(record.days_to_disclosure),
            "daysToPayment": str(record.days_to_payment),
            "claimAmount": str(record.claim_amount_cents),
            "settlementAmount": str(record.settlement_amount_cents),
            "approved": str(record.approved),
            "lowballReasoningProvided": str(record.lowball_reasoning_provided),
        }
        # Run traditional check in parallel
        trad = self._traditional.check(record)

        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "input.json"
            proof_path = Path(td) / "proof.json"
            public_path = Path(td) / "public.json"
            input_path.write_text(json.dumps(circuit_input))
            try:
                subprocess.run(
                    [
                        "snarkjs", "groth16", "fullprove",
                        str(input_path), str(wasm), str(zkey),
                        str(proof_path), str(public_path),
                    ],
                    check=True, capture_output=True, timeout=60,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                    FileNotFoundError) as e:
                logger.warning("snarkjs prove failed (%s); falling back to stub", e)
                return self._prove_stub(record)

            proof_obj = json.loads(proof_path.read_text())
            public_arr = json.loads(public_path.read_text())

        latency_ms = (time.perf_counter() - started) * 1000.0
        verified = (public_arr[0] == "1") if public_arr else False
        abbr = CODE_TO_ABBR.get(record.jurisdiction_code, "?")
        claim_type_name = next(
            (k for k, v in CLAIM_TYPE_CODES.items() if v == record.claim_type_code),
            "unknown"
        )
        statement = (
            f"Compliance proof {'VERIFIED' if verified else 'FAILED'} "
            f"for jurisdiction={abbr}; claim_type={claim_type_name}; "
            f"approved={record.approved}; "
            f"checks={trad.get('checks')}."
        )
        return ComplianceProofResult(
            verified=verified,
            statement=statement,
            proof_type="groth16",
            jurisdiction=abbr,
            claim_type=claim_type_name,
            compliance_root=str(compliance_root),
            proof=json.dumps(proof_obj),
            public_signals={"_circuit_public": public_arr,
                             "jurisdiction": abbr,
                             "claim_type": claim_type_name},
            prover_latency_ms=latency_ms,
            checks=trad.get("checks", {}),
        )

    def _verify_real(self, proof: dict[str, Any] | str,
                     public_signals: dict[str, Any] | list[Any]) -> ComplianceVerifyResult:
        """Verify a real Groth16 proof via snarkjs."""
        started = time.perf_counter()
        vkey = self.circuit_dir / "keys" / "compliance_verification_key.json"
        try:
            proof_obj = json.loads(proof) if isinstance(proof, str) else proof
        except (json.JSONDecodeError, TypeError):
            return ComplianceVerifyResult(
                verified=False, verifier="groth16",
                verification_key=str(vkey), latency_ms=0.0,
                reason="Malformed proof: cannot deserialize JSON.",
            )
        # Reconstruct public signals array
        if isinstance(public_signals, dict) and "_circuit_public" in public_signals:
            public_arr = public_signals["_circuit_public"]
        elif isinstance(public_signals, list):
            public_arr = public_signals
        else:
            is_compliant = "1" if public_signals.get("verified") else "0"
            public_arr = [is_compliant,
                          str(public_signals.get("compliance_root", "0")),
                          str(public_signals.get("claim_type", "0"))]
        with tempfile.TemporaryDirectory() as td:
            p_path = Path(td) / "proof.json"
            pub_path = Path(td) / "public.json"
            p_path.write_text(json.dumps(proof_obj))
            pub_path.write_text(json.dumps(public_arr))
            try:
                result = subprocess.run(
                    ["snarkjs", "groth16", "verify",
                     str(vkey), str(pub_path), str(p_path)],
                    check=False, capture_output=True, timeout=15,
                )
                verified = (result.returncode == 0)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.warning("snarkjs verify failed: %s", e)
                verified = False
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ComplianceVerifyResult(
            verified=verified, verifier="groth16",
            verification_key=str(vkey), latency_ms=latency_ms,
            reason="ok" if verified else "snarkjs verification failed",
        )

    # ------------------------------------------------------------------ #
    # Stub prover/verifier (fallback when circuit not compiled)
    # ------------------------------------------------------------------ #
    def _prove_stub(self, record: ComplianceClaimRecord) -> ComplianceProofResult:
        """Fallback stub prover. Re-evaluates the constraints in Python
        and returns a structured result. NOT cryptographically secure."""
        started = time.perf_counter()
        trad = self._traditional.check(record)
        verified = bool(trad.get("compliant"))
        abbr = CODE_TO_ABBR.get(record.jurisdiction_code, "?")
        claim_type_name = next(
            (k for k, v in CLAIM_TYPE_CODES.items() if v == record.claim_type_code),
            "unknown"
        )
        # Stub "proof" = SHA-256 of the public inputs + verified flag.
        # This lets the stub verifier re-derive and compare, similar to
        # the policy_validity stub pattern.
        compliance_root = self._compute_compliance_root_stub(record)
        public_signals = {
            "jurisdiction": abbr,
            "claim_type": claim_type_name,
            "compliance_root": compliance_root,
            "verified": verified,
        }
        proof_payload = json.dumps({
            "circuit": "compliance_verification.circom",
            "public_signals": public_signals,
        }, sort_keys=True).encode()
        proof_hash = hashlib.sha256(proof_payload).hexdigest()
        proof = f"zkpc:{proof_hash}" + "0" * max(0, 200 - len(f"zkpc:{proof_hash}"))
        latency_ms = (time.perf_counter() - started) * 1000.0
        statement = (
            f"Compliance proof {'VERIFIED' if verified else 'FAILED'} "
            f"(stub) for jurisdiction={abbr}; claim_type={claim_type_name}; "
            f"approved={record.approved}; checks={trad.get('checks')}."
        )
        return ComplianceProofResult(
            verified=verified,
            statement=statement,
            proof_type="groth16-stub",
            jurisdiction=abbr,
            claim_type=claim_type_name,
            compliance_root=compliance_root,
            proof=proof,
            public_signals=public_signals,
            prover_latency_ms=latency_ms,
            checks=trad.get("checks", {}),
        )

    def _verify_stub(self, proof: Any, public_signals: Any) -> ComplianceVerifyResult:
        """Stub verifier — re-derives the expected proof hash and compares."""
        started = time.perf_counter()
        if not isinstance(proof, str) or not proof.startswith("zkpc:"):
            return ComplianceVerifyResult(
                verified=False, verifier="groth16-stub",
                verification_key="vk:compliance.v1", latency_ms=0.0,
                reason="Malformed proof: must start with 'zkpc:'.",
            )
        # Reconstruct expected public signals
        if isinstance(public_signals, dict):
            ps = public_signals
        else:
            return ComplianceVerifyResult(
                verified=False, verifier="groth16-stub",
                verification_key="vk:compliance.v1", latency_ms=0.0,
                reason="public_signals must be a dict for stub verification.",
            )
        expected_payload = json.dumps({
            "circuit": "compliance_verification.circom",
            "public_signals": ps,
        }, sort_keys=True).encode()
        expected_hash = hashlib.sha256(expected_payload).hexdigest()
        expected_proof = f"zkpc:{expected_hash}"
        supplied = proof[: len(expected_proof)]
        verified = (supplied == expected_proof)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ComplianceVerifyResult(
            verified=verified, verifier="groth16-stub",
            verification_key="vk:compliance.v1", latency_ms=latency_ms,
            reason="ok" if verified else "Proof hash mismatch — tampered or malformed.",
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _compute_compliance_root_stub(self, record: ComplianceClaimRecord) -> str:
        """Compute a SHA-256-based complianceRoot.

        Production replaces this with Poseidon(claimRecordCommitment, salt)
        via circomlibjs. The stub uses SHA-256 so unit tests can run
        without the Poseidon JS library.
        """
        payload = json.dumps({
            "claim_record_commitment": record.claim_record_commitment,
            "salt": record.salt,
        }, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


# ===========================================================================
# Convenience: build a ComplianceClaimRecord from a context dict
# ===========================================================================
def build_record_from_context(
    ctx: dict[str, Any],
    *,
    approved: int = 1,
    days_to_acknowledge: int = 5,
    days_to_disclosure: int = 7,
    days_to_payment: int = 10,
    settlement_ratio: float = 1.0,
    lowball_reasoning_provided: int = 0,
    salt: int = 42,
) -> ComplianceClaimRecord:
    """Build a :class:`ComplianceClaimRecord` from a claim context dict.

    The context dict is the one passed between agents in the state
    machine. This helper extracts the jurisdiction, claim type, amount,
    and policy information and constructs the private inputs needed by
    the compliance circuit.

    The defaults for ``days_to_acknowledge`` etc. are conservative
    (5 days ack, 7 days disclosure, 10 days payment) and satisfy every
    state's regulatory deadlines for a fully-compliant claim.
    """
    inputs = ctx.get("zkp_policy_inputs", {}) or {}
    jurisdiction = inputs.get("jurisdiction", "CA")
    reg = STATE_REGULATIONS.get(jurisdiction, STATE_REGULATIONS["CA"])
    claim_type = (ctx.get("claim_type") or "property_damage").lower()
    claim_type_code = CLAIM_TYPE_CODES.get(claim_type, 1)
    amount = float(ctx.get("claim", {}).get("amount", 0) or 0)
    claim_amount_cents = int(round(amount * 100))
    settlement_cents = int(round(claim_amount_cents * settlement_ratio))
    # Build a deterministic claim record commitment (stub: SHA-256 of
    # the claim_id + amount + dates).
    claim_id = ctx.get("claim", {}).get("claim_id", "unknown")
    commitment_payload = json.dumps({
        "claim_id": claim_id,
        "amount": claim_amount_cents,
        "jurisdiction": jurisdiction,
    }, sort_keys=True).encode()
    commitment = int.from_bytes(hashlib.sha256(commitment_payload).digest()[:8],
                                 "big")
    return ComplianceClaimRecord(
        claim_record_commitment=commitment,
        salt=salt,
        jurisdiction_code=reg.code,
        claim_type_code=claim_type_code,
        days_to_acknowledge=days_to_acknowledge,
        days_to_disclosure=days_to_disclosure,
        days_to_payment=days_to_payment,
        claim_amount_cents=claim_amount_cents,
        settlement_amount_cents=settlement_cents,
        approved=approved,
        lowball_reasoning_provided=lowball_reasoning_provided,
    )


# ===========================================================================
# Module-level singleton for convenience
# ===========================================================================
_DEFAULT_PROVER: Optional[ComplianceProver] = None


def get_default_prover() -> ComplianceProver:
    """Return a process-wide default ComplianceProver instance."""
    global _DEFAULT_PROVER
    if _DEFAULT_PROVER is None:
        _DEFAULT_PROVER = ComplianceProver()
    return _DEFAULT_PROVER
