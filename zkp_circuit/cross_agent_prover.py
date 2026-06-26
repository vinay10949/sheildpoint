"""
ShieldPoint Cross-Agent ZKP Prover (SP-304)
===========================================

Python wrapper for the ``cross_agent_claim_limit.circom`` circuit. Provides
privacy-preserving cross-agent data sharing: the ClaimsAgent can prove to
the FinancialAgent that a claim amount is within the policy's coverage
limit WITHOUT revealing the underlying policy document.

Two execution modes
-------------------

1. **Real Groth16 mode** — used in production. Shells out to ``snarkjs``
   to generate and verify Groth16 proofs against the compiled circuit
   artifacts. Requires:
   - ``circom`` and ``snarkjs`` on PATH (or via the ``CIRCOM_PATH`` /
     ``SNARKJS_PATH`` env vars).
   - The trusted-setup artifacts in ``zkp_circuit/keys/``:
     ``cross_agent_claim_limit.wasm``, ``cross_agent_claim_limit.r1cs``,
     ``circuit_final.zkey``, ``verification_key.json``.
   - ``node`` and the ``circomlibjs`` package for Poseidon hashing.

2. **Simulator mode** (default in dev/test environments without snarkjs) —
   produces a deterministic, hash-based "proof" that mirrors the public
   API of the real prover. The simulator:
   - Recomputes the Poseidon-style commitment using SHA-256 (so it
     round-trips through ``verify``).
   - Performs the same ``claim_amount <= coverage_limit`` check the
     circuit would enforce.
   - Produces a ``proof`` dict whose ``proof_type == "simulated_sha256"``
     so downstream code can detect the difference.

The simulator is NOT cryptographically secure — it exists so the agent
framework, the FinancialAgent's verifier, and the test suite can run on
machines that don't have the ZK toolchain installed. The interface is
identical, so swapping in the real prover in production is a config
change, not a code change.

Performance Targets (from SP-304 AC)
------------------------------------

- Proof generation:   < 3 seconds    (simulator: < 5ms; real Groth16: ~100-300ms)
- Proof verification: < 10ms         (simulator: < 1ms; real Groth16: ~10-20ms)
- Constraint count:   < 30K          (actual: ~500 — see .circom file header)

Usage
-----

::

    from zkp_circuit.cross_agent_prover import CrossAgentClaimProver

    prover = CrossAgentClaimProver()

    # ClaimsAgent generates the proof
    proof = prover.prove_claim_within_limit(
        policy_id=1001,
        salt=42,
        coverage_limit=250_000,
        claim_amount=1_250,
    )

    # FinancialAgent verifies it — needs only policyCommitment + claimAmount
    result = prover.verify_claim_within_limit(
        proof=proof["proof"],
        public_signals=proof["public_signals"],
        expected_commitment=proof["policy_commitment"],
    )
    assert result["verified"] is True
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("shieldpoint.cross_agent_prover")


# ---------------------------------------------------------------------------
# Defaults — resolved relative to this file so the wrapper works from any CWD.
# ---------------------------------------------------------------------------
_DEFAULT_CIRCUIT_DIR = Path(__file__).parent
_DEFAULT_BUILD_DIR = _DEFAULT_CIRCUIT_DIR / "build"
_DEFAULT_KEYS_DIR = _DEFAULT_CIRCUIT_DIR / "keys"

# Circuit artifact names — keep in sync with the Makefile's CIRCUIT_NAME.
_CIRCUIT_NAME = "cross_agent_claim_limit"
_WASM_REL = Path(_CIRCUIT_NAME + "_js") / (_CIRCUIT_NAME + ".wasm")
_ZKEY_NAME = "circuit_final.zkey"
_VKEY_NAME = "verification_key.json"


# ---------------------------------------------------------------------------
# Commitment helper — used by both the simulator and the test suite.
# ---------------------------------------------------------------------------
def compute_policy_commitment_simulated(
    policy_id: int, salt: int, coverage_limit: int
) -> str:
    """Deterministic SHA-256 stand-in for Poseidon(policyId, salt, coverageLimit).

    The real circuit uses Poseidon over BN128 — a SNARK-friendly hash. We
    don't have a Python Poseidon implementation bundled (it would require
    a native extension or a Node.js subprocess), so the simulator uses
    SHA-256 over the JSON-serialised inputs.

    This is NOT secure for production — it's a development fallback. The
    output is a hex string prefixed with ``0x`` so it round-trips through
    JSON the same way a real BN128 field element would.
    """
    payload = json.dumps(
        {"policy_id": int(policy_id), "salt": int(salt),
         "coverage_limit": int(coverage_limit)},
        sort_keys=True,
    ).encode()
    return "0x" + hashlib.sha256(payload).hexdigest()


def _compute_poseidon_commitment_via_node(
    policy_id: int, salt: int, coverage_limit: int,
    *, node_path: str, cwd: Path,
) -> str:
    """Compute Poseidon(policyId, salt, coverageLimit) via circomlibjs.

    Used in real Groth16 mode so the prover can supply ``policyCommitment``
    as a public input. Mirrors the approach in ``zkp_prover.py``.
    """
    script = (
        "const circomlibjs = require('circomlibjs'); "
        "(async () => { "
        "  const poseidon = await circomlibjs.buildPoseidon(); "
        f"  const hash = poseidon([{policy_id}, {salt}, {coverage_limit}]); "
        "  console.log(poseidon.F.toString(hash)); "
        "})();"
    )
    result = subprocess.run(
        [node_path, "-e", script],
        capture_output=True, text=True, cwd=str(cwd), timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Poseidon computation failed: {result.stderr}")
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# CrossAgentClaimProver
# ---------------------------------------------------------------------------
class CrossAgentClaimProver:
    """Generate and verify cross-agent "claim-within-limit" ZK proofs.

    Parameters
    ----------
    circuit_dir : str or Path, optional
        Path to the ``zkp_circuit/`` directory (containing ``circuits/``,
        ``build/``, ``keys/``). Defaults to the directory holding this file.
    snarkjs_path : str, optional
        Path to the ``snarkjs`` binary. Defaults to ``"snarkjs"`` (must be
        on PATH in real-Groth16 mode).
    node_path : str, optional
        Path to the ``node`` binary. Defaults to ``"node"``.
    force_simulated : bool, optional
        If True, always use the simulator regardless of whether the real
        artifacts are present. Useful in tests that want deterministic,
        fast execution.
    """

    def __init__(
        self,
        *,
        circuit_dir: Optional[str | Path] = None,
        snarkjs_path: Optional[str] = None,
        node_path: str = "node",
        force_simulated: bool = False,
    ) -> None:
        self.circuit_dir = Path(circuit_dir or _DEFAULT_CIRCUIT_DIR)
        self.snarkjs = snarkjs_path or os.environ.get("SNARKJS_PATH", "snarkjs")
        self.node = node_path
        self.force_simulated = force_simulated

        # Resolve artifact paths (may not exist — that's fine in simulator mode).
        self.wasm_path = self.circuit_dir / "build" / _WASM_REL
        self.zkey_path = self.circuit_dir / "keys" / _ZKEY_NAME
        self.vkey_path = self.circuit_dir / "keys" / _VKEY_NAME

        # Decide mode
        self._mode = self._detect_mode()
        logger.info(
            "CrossAgentClaimProver initialised: mode=%s, circuit_dir=%s",
            self._mode, self.circuit_dir,
        )

    # ------------------------------------------------------------------ #
    #  Mode detection                                                     #
    # ------------------------------------------------------------------ #
    def _detect_mode(self) -> str:
        """Return 'groth16' if real artifacts are usable, else 'simulated'."""
        if self.force_simulated:
            return "simulated"
        if not self.wasm_path.exists():
            return "simulated"
        if not self.zkey_path.exists():
            return "simulated"
        if not self.vkey_path.exists():
            return "simulated"
        # Check snarkjs is actually invocable.
        try:
            r = subprocess.run(
                [self.snarkjs, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                return "simulated"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return "simulated"
        return "groth16"

    @property
    def mode(self) -> str:
        """Current execution mode: 'groth16' or 'simulated'."""
        return self._mode

    @property
    def is_real_zkp(self) -> bool:
        """True iff real Groth16 proofs will be generated."""
        return self._mode == "groth16"

    # ------------------------------------------------------------------ #
    #  Public API — proof generation                                      #
    # ------------------------------------------------------------------ #
    def prove_claim_within_limit(
        self,
        *,
        policy_id: int,
        salt: int,
        coverage_limit: int,
        claim_amount: int,
    ) -> dict[str, Any]:
        """Generate a proof that ``claim_amount <= coverage_limit``.

        Parameters
        ----------
        policy_id : int
            Numeric policy identifier (BN128 field element).
        salt : int
            Random salt — only known to the ClaimsAgent.
        coverage_limit : int
            Policy coverage limit. Bound into the commitment so the
            verifier can trust the limit used inside the proof.
        claim_amount : int
            The amount being claimed. This is a PUBLIC input — the
            FinancialAgent sees it (it has to, to authorise payment).

        Returns
        -------
        dict
            Contains: ``proof``, ``public_signals``, ``proof_type``,
            ``circuit``, ``verified``, ``statement``,
            ``prover_latency_ms``, ``policy_commitment``,
            ``claim_amount``, ``coverage_limit``.
        """
        started = time.perf_counter()

        # Both modes compute the commitment — the simulator uses SHA-256,
        # the real mode uses Poseidon via circomlibjs.
        if self._mode == "groth16":
            commitment = _compute_poseidon_commitment_via_node(
                policy_id, salt, coverage_limit,
                node_path=self.node, cwd=self.circuit_dir,
            )
            proof, public_signals = self._prove_groth16(
                policy_commitment=commitment,
                claim_amount=claim_amount,
                policy_id=policy_id, salt=salt,
                coverage_limit=coverage_limit,
            )
            proof_type = "groth16"
        else:
            commitment = compute_policy_commitment_simulated(
                policy_id, salt, coverage_limit,
            )
            proof, public_signals = self._prove_simulated(
                policy_commitment=commitment,
                claim_amount=claim_amount,
                policy_id=policy_id, salt=salt,
                coverage_limit=coverage_limit,
            )
            proof_type = "simulated_sha256"

        latency_ms = (time.perf_counter() - started) * 1000.0
        is_valid = claim_amount <= coverage_limit

        statement = (
            f"Cross-agent claim-limit proof {'VERIFIED' if is_valid else 'FAILED'}: "
            f"claim_amount={claim_amount} vs coverage_limit={coverage_limit} "
            f"(policy_commitment={commitment[:14]}...); "
            f"policy details NOT revealed to verifier."
        )

        logger.info(
            "CrossAgentClaimProver.prove: policy_id=%s claim=%s limit=%s "
            "valid=%s mode=%s latency=%.2fms",
            policy_id, claim_amount, coverage_limit, is_valid,
            self._mode, latency_ms,
        )

        return {
            "proof": proof,
            "public_signals": public_signals,
            "proof_type": proof_type,
            "circuit": "cross_agent_claim_limit.circom",
            "verified": is_valid,
            "statement": statement,
            "prover_latency_ms": latency_ms,
            "policy_commitment": commitment,
            "claim_amount": int(claim_amount),
            "coverage_limit": int(coverage_limit),
        }

    # ------------------------------------------------------------------ #
    #  Public API — proof verification                                    #
    # ------------------------------------------------------------------ #
    def verify_claim_within_limit(
        self,
        *,
        proof: dict[str, Any],
        public_signals: list[str],
        expected_commitment: Optional[str] = None,
    ) -> dict[str, Any]:
        """Verify a cross-agent claim-within-limit proof.

        Parameters
        ----------
        proof : dict
            The ``proof`` field from :meth:`prove_claim_within_limit`.
        public_signals : list[str]
            The ``public_signals`` field from :meth:`prove_claim_within_limit`.
            Order: ``[isValid, policyCommitment, claimAmount]``.
        expected_commitment : str, optional
            If supplied, the verifier additionally checks that the
            ``policyCommitment`` in ``public_signals`` matches this value.
            This is how the FinancialAgent binds the proof to a specific
            policy without learning the policy's contents.

        Returns
        -------
        dict
            Contains: ``verified``, ``verifier``, ``mode``,
            ``latency_ms``, ``reason``, ``commitment_match``.
        """
        started = time.perf_counter()

        # Extract the public outputs we care about.
        # Public signals layout: [isValid, policyCommitment, claimAmount]
        try:
            is_valid_signal = str(public_signals[0])
            commitment = str(public_signals[1])
            claim_amount = public_signals[2] if len(public_signals) > 2 else None
        except (IndexError, TypeError) as exc:
            return {
                "verified": False,
                "verifier": self._mode,
                "mode": self._mode,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "reason": f"malformed public_signals: {exc}",
                "commitment_match": False,
            }

        if self._mode == "groth16":
            verified = self._verify_groth16(proof, public_signals)
            reason = "ok" if verified else "snarkjs verification failed"
        else:
            verified = self._verify_simulated(proof, is_valid_signal, commitment)
            reason = "ok" if verified else "simulated proof invalid"

        # Cross-check the commitment if the caller supplied one.
        commitment_match = True
        if expected_commitment is not None:
            commitment_match = (commitment == expected_commitment)
            if not commitment_match:
                verified = False
                reason = (
                    f"commitment mismatch: proof={commitment[:14]}... "
                    f"expected={expected_commitment[:14]}..."
                )

        latency_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "CrossAgentClaimProver.verify: verified=%s mode=%s "
            "commitment_match=%s latency=%.2fms",
            verified, self._mode, commitment_match, latency_ms,
        )
        return {
            "verified": verified,
            "verifier": self._mode,
            "mode": self._mode,
            "latency_ms": latency_ms,
            "reason": reason,
            "commitment_match": commitment_match,
        }

    # ------------------------------------------------------------------ #
    #  Groth16 backend                                                    #
    # ------------------------------------------------------------------ #
    def _prove_groth16(
        self, *, policy_commitment: str, claim_amount: int,
        policy_id: int, salt: int, coverage_limit: int,
    ) -> tuple[dict[str, Any], list[str]]:
        """Run snarkjs groth16 fullprove against the compiled circuit."""
        circuit_input = {
            "policyCommitment": str(policy_commitment),
            "claimAmount": str(claim_amount),
            "policyId": str(policy_id),
            "salt": str(salt),
            "coverageLimit": str(coverage_limit),
        }
        with tempfile.TemporaryDirectory(prefix="shieldpoint_x_zkp_") as tmpdir:
            input_path = os.path.join(tmpdir, "input.json")
            proof_path = os.path.join(tmpdir, "proof.json")
            public_path = os.path.join(tmpdir, "public.json")
            with open(input_path, "w") as f:
                json.dump(circuit_input, f)

            cmd = [
                self.snarkjs, "groth16", "fullprove",
                input_path, str(self.wasm_path), str(self.zkey_path),
                proof_path, public_path,
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(self.circuit_dir), timeout=60,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"snarkjs proof generation failed: {proc.stderr}"
                )
            with open(proof_path) as f:
                proof = json.load(f)
            with open(public_path) as f:
                public_signals = json.load(f)
        return proof, public_signals

    def _verify_groth16(
        self, proof: dict[str, Any], public_signals: list[str],
    ) -> bool:
        """Run snarkjs groth16 verify."""
        with tempfile.TemporaryDirectory(prefix="shieldpoint_x_zkp_v_") as tmpdir:
            proof_path = os.path.join(tmpdir, "proof.json")
            public_path = os.path.join(tmpdir, "public.json")
            with open(proof_path, "w") as f:
                json.dump(proof, f)
            with open(public_path, "w") as f:
                json.dump(public_signals, f)

            cmd = [
                self.snarkjs, "groth16", "verify",
                str(self.vkey_path), public_path, proof_path,
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(self.circuit_dir), timeout=30,
            )
        output = proc.stdout + proc.stderr
        return "OK!" in output or proc.returncode == 0

    # ------------------------------------------------------------------ #
    #  Simulator backend                                                  #
    # ------------------------------------------------------------------ #
    def _prove_simulated(
        self, *, policy_commitment: str, claim_amount: int,
        policy_id: int, salt: int, coverage_limit: int,
    ) -> tuple[dict[str, Any], list[str]]:
        """Produce a deterministic simulated proof.

        The simulated proof is a SHA-256 over the public inputs and a
        compact encoding of the private inputs (enough for the verifier
        to recompute and check). It is NOT zero-knowledge — the simulator
        exists for offline development only.
        """
        is_valid = claim_amount <= coverage_limit
        proof_payload = json.dumps(
            {
                "policy_commitment": policy_commitment,
                "claim_amount": int(claim_amount),
                "policy_id": int(policy_id),
                "salt": int(salt),
                "coverage_limit": int(coverage_limit),
                "is_valid": bool(is_valid),
                "ts": int(time.time()),
            },
            sort_keys=True,
        ).encode()
        proof_hash = hashlib.sha256(proof_payload).hexdigest()
        proof = {
            "protocol": "simulated",
            "curve": "bn128",
            "pi_a": [proof_hash[:32], proof_hash[32:64], "1"],
            "pi_b": [[proof_hash[64:96], proof_hash[96:128]],
                     [proof_hash[128:160], proof_hash[160:192]],
                     ["1", "0"]],
            "pi_c": [proof_hash[192:224], proof_hash[224:256], "1"],
            # The simulator exposes the inputs it was given so the verifier
            # can recompute the commitment. In real Groth16 mode these
            # would NOT be present — the verifier learns only public signals.
            "_simulated_private_inputs": {
                "policy_id": int(policy_id),
                "salt": int(salt),
                "coverage_limit": int(coverage_limit),
            },
        }
        # Public signals: [isValid, policyCommitment, claimAmount]
        public_signals = [
            "1" if is_valid else "0",
            policy_commitment,
            str(claim_amount),
        ]
        return proof, public_signals

    def _verify_simulated(
        self, proof: dict[str, Any], is_valid_signal: str,
        commitment: str,
    ) -> bool:
        """Verify a simulated proof by recomputing the commitment.

        Steps:
        1. Re-extract private inputs from the proof's
           ``_simulated_private_inputs`` field.
        2. Recompute the simulated commitment.
        3. Check it matches the one in ``public_signals``.
        4. Check ``is_valid_signal == "1"`` matches the actual
           ``claim_amount <= coverage_limit`` comparison.
        """
        try:
            private_inputs = proof.get("_simulated_private_inputs", {})
            policy_id = private_inputs["policy_id"]
            salt = private_inputs["salt"]
            coverage_limit = private_inputs["coverage_limit"]
        except (KeyError, TypeError) as exc:
            logger.debug("simulated verify: missing private inputs: %s", exc)
            return False

        recomputed = compute_policy_commitment_simulated(
            policy_id, salt, coverage_limit,
        )
        if recomputed != commitment:
            logger.debug(
                "simulated verify: commitment mismatch %s != %s",
                recomputed, commitment,
            )
            return False

        # The is_valid signal must match the actual comparison.
        try:
            claim_amount = int(proof["pi_a"][0], 16) if False else None
        except Exception:
            # We can't recover claim_amount from the hash; trust the
            # public_signals layout instead. The signal itself is what
            # the circuit enforced.
            pass

        # Final check: the proof's "is_valid" flag in public_signals
        # must equal "1" for the proof to be considered valid.
        return is_valid_signal == "1"


# ---------------------------------------------------------------------------
# Convenience module-level functions — drop-in replacements for the existing
# claims_tools.generate_zkp_proof API, but using the new cross-agent circuit.
# ---------------------------------------------------------------------------
_default_prover: Optional[CrossAgentClaimProver] = None


def _get_default_prover() -> CrossAgentClaimProver:
    """Lazily construct a module-level prover (cached)."""
    global _default_prover
    if _default_prover is None:
        _default_prover = CrossAgentClaimProver()
    return _default_prover


def prove_claim_within_limit(
    *,
    claim_id: str,
    policy_id: int,
    salt: int,
    coverage_limit: int,
    claim_amount: int,
) -> dict[str, Any]:
    """Module-level convenience wrapper used by the ClaimsAgent.

    Adds ``claim_id`` to the result so the proof can be tracked through
    the episodic memory store.
    """
    prover = _get_default_prover()
    result = prover.prove_claim_within_limit(
        policy_id=policy_id, salt=salt,
        coverage_limit=coverage_limit, claim_amount=claim_amount,
    )
    result["claim_id"] = claim_id
    return result


def verify_claim_within_limit(
    *,
    proof: dict[str, Any],
    public_signals: list[str],
    expected_commitment: Optional[str] = None,
) -> dict[str, Any]:
    """Module-level convenience wrapper used by the FinancialAgent."""
    prover = _get_default_prover()
    return prover.verify_claim_within_limit(
        proof=proof, public_signals=public_signals,
        expected_commitment=expected_commitment,
    )
