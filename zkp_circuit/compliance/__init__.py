"""ShieldPoint ZKP Compliance Verification package.

Public API:
- :class:`ComplianceProver` — main prover/verifier class
- :class:`TraditionalComplianceChecker` — parallel non-ZKP path
- :class:`ComplianceClaimRecord` — private inputs dataclass
- :data:`STATE_REGULATIONS`, :data:`CODE_TO_ABBR`, :data:`CLAIM_TYPE_CODES`
- :func:`build_record_from_context` — convenience builder
"""

from .compliance_prover import (
    CLAIM_TYPE_CODES,
    CODE_TO_ABBR,
    STATE_REGULATIONS,
    ComplianceClaimRecord,
    ComplianceProofResult,
    ComplianceProver,
    ComplianceVerifyResult,
    StateRegulation,
    TraditionalComplianceChecker,
    build_record_from_context,
    get_default_prover,
)

__all__ = [
    "ComplianceProver",
    "TraditionalComplianceChecker",
    "ComplianceClaimRecord",
    "ComplianceProofResult",
    "ComplianceVerifyResult",
    "StateRegulation",
    "STATE_REGULATIONS",
    "CODE_TO_ABBR",
    "CLAIM_TYPE_CODES",
    "build_record_from_context",
    "get_default_prover",
]
